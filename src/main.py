import torch
import random
import argparse
import pickle
import os
import logging
import time
import numpy as np
import torch.backends.cudnn as cudnn
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model_utils.pre_optimizer_utils import add_optimizer_args, get_total_steps, configure_optimizers
from dataset_utils.pre_img_seq import load_data, add_datasets_args
from dataset_utils.img_seq_universal_datamodule import UniversalDataModule
from ckpt_utils.img_seq_rec_universal_ckpt import UniversalCheckpoint 
from torch.utils.data._utils.collate import default_collate
from models.img_seq_rec import Preimgseqrec, Linear_pred
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


seq_nums = {'Sports': 207901, 'Beauty': 143514, 'Toys': 121285}


def pop_item(datapkl, ratio):
    items = []
    for seq in datapkl:
        items += list(seq.values())[0]
    num_dict = Counter(items)
    pop_item = [i[0] for i in num_dict.most_common(int(len(num_dict)*(1-ratio)))]
    
    return pop_item

def pop_item_interaction(datapkl, ratio):
    items = []
    for seq in datapkl:
        items += list(seq.values())[0]
    num_dict = Counter(items)
    all_intr_nums = sum(list(num_dict.values()))
    pop_inter = int(ratio*all_intr_nums)
    pop_item = []
    count_acc = 0
    for item, num in zip(num_dict.keys(), num_dict.values()):
        count_acc += num
        if count_acc < pop_inter:
            pop_item.append(item)
        else:
            break
    return pop_item
        
        
class Collator():
    def __init__(self, args, pop_items):
        self.len_seq = args.seq_len
        # with open('/R2Rec/data/merge/' + args.datasets_name+'_asin2id.pickle', 'rb') as f:
        with open('/R2Rec/data/' + args.datasets_name+'/asin2id.pickle', 'rb') as f:
        
            self.id_asin_dict = pickle.load(f)
        self.pop_item = [self.id_asin_dict[item]+1 for item in pop_items]
        self.tail_item = [self.id_asin_dict[item]+1 for item in self.id_asin_dict if item not in pop_items]
     
    def __call__(self, inputs):
        examples = []
        for idx, input_temp in enumerate(inputs):
            input_temp = list(input_temp.values())[0]
            example = {}
            seqs_temp = input_temp[-(self.len_seq+1):]
            seq_temp = seqs_temp[:-1]
            label = seqs_temp[-1]
            seq_temp_id = [self.id_asin_dict[i]+1 for i in seq_temp]  ## +1 as padding is 0
            label_id = self.id_asin_dict[label] + 1
            seq_temp_id_pad = [0] * (self.len_seq - len(seq_temp_id)) + seq_temp_id
            mask_temp = [0] * (self.len_seq - len(seq_temp_id)) + len(seq_temp_id) * [1]
            example['seq_id'] =  seq_temp_id_pad
            example['mask_seq'] = mask_temp 
            example['label_id'] = label_id
            example['userid'] = idx
            
            if label_id not in self.pop_item:
                example['cold_item_flag'] = 1
                example['plenty'] = 1.0
            else:
                example['cold_item_flag'] = 0
                example['plenty'] = 0.0
            examples.append(example)
        return default_collate(examples)
               

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string)


def cal_recall(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def recalls_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_recall(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['Recall@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def calculate_metrics(scores, labels, metric_ks):
    metrics = recalls_and_ndcgs_k(scores, labels, metric_ks)
    return metrics


class TailImgRec(LightningModule):
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('PretrainingSeq img Rec')
        parser.add_argument('--seq_len', type=int, default=10)
        parser.add_argument('--hidden_size', type=int, default=768)
        parser.add_argument('--attn_head', type=int, default=16)
        parser.add_argument('--n_blocks', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.8)
        parser.add_argument('--info', type=str, default=None)
        # parser.add_argument('--emb_img_model', type=str, default=None)
        parser.add_argument('--item_nums', type=int, default=None)
        parser.add_argument('--tau', type=float, default=0.1)
        return parent_parser
    
    def __init__(self, args, tail_items, logger):
        super().__init__()
        
        ## Title embedding
        # emb = torch.load('/R2Rec/data/'+args.datasets_name+'/title_emb_clip_vit_large_patch14_len10.pt')
        
        ### Img embedding Pretrain
        ## emb = torch.load('/R2Rec/data/merge/'+args.datasets_name+'_img_emb_clip_vit_large_patch14.pt')
        
        
        emb = torch.load('/R2Rec/data/'+args.datasets_name+'/img_emb_clip_vit_large_patch14.pt')
        pad_emb = torch.mean(emb, dim=0)
        embs = torch.cat([pad_emb.unsqueeze(0), emb], dim=0)
        # ## Reembedding tail item
        embs[tail_items] = torch.nn.Embedding(len(tail_items), 768).weight
        self.item_emb_img = torch.nn.Embedding.from_pretrained(embs)

        ### ID Embedding
        # self.item_emb_img = torch.nn.Embedding(args.item_nums+1, 768)
        
        self.item_emb_img.weight.requires_grad = True
        
        self.pre_imgseqrec_model = Preimgseqrec(args)
        self.pred_layer = Linear_pred(args, self.item_emb_img.weight.shape[0])
        self.loss_ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_ce_raw = nn.CrossEntropyLoss()
        self.logger_save = logger
        self.tau = args.tau
        self.plenty = torch.tensor([1 for i in range(self.item_emb_img.weight.shape[0])]).float().to(torch.cuda.current_device())
        self.save_hyperparameters(args)
    
    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))
    
    def configure_optimizers(self):
        return configure_optimizers(self)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def training_step(self, batch, batch_idx):
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_img = torch.stack(batch['seq_id'], dim=1)
        seq_rep_img = self.item_emb_img(seq_img)
        rep_seq = self.pre_imgseqrec_model(seq_rep_img, masks)
        prod = self.pred_layer(rep_seq)
        """
        ## enty for items
        plenty_term = torch.softmax(self.plenty/self.tau, dim=0)
        _, ind_pre = torch.topk(prod, k=5)
        count = (1-(batch['label_id'].long().unsqueeze(1) == ind_pre).float().sum(dim=-1).to(prod.device))
        tailitem_seq_id = torch.where(batch['cold_item_flag']==1)
        self.plenty[batch['label_id'][tailitem_seq_id]] += count[tailitem_seq_id]
        loss = self.loss_ce_raw(prod*plenty_term.unsqueeze(0), batch['label_id'].long())
        """
        
        ## plenty for sequences
        loss = self.loss_ce(prod, batch['label_id'].long())
        if batch['cold_item_flag'].sum() == 0:
            plenty_term = 1 
        else:
            plenty_term = torch.softmax(batch['plenty']/self.tau, dim=0)
            
        #     _, ind_pre = torch.topk(prod, k=5)
        #     count = (batch['label_id'].long().unsqueeze(1) == ind_pre).float().sum(dim=-1).to(prod.device)
        #     batch['plenty'] += batch['cold_item_flag'] * count
        loss = (plenty_term * loss).mean()
        
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        if self.trainer.global_rank == 0 and self.global_step == 100:
            report_memory('Seq rec')
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        self.pre_imgseqrec_model.eval()
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_img = torch.stack(batch['seq_id'], dim=1)
        seq_rep_img = self.item_emb_img(seq_img)
        rep_seq = self.pre_imgseqrec_model(seq_rep_img, masks)
        prod = self.pred_layer(rep_seq)
        cold_idx = torch.where(batch['cold_item_flag']==1)
        metrics = calculate_metrics(prod, batch['label_id'].unsqueeze(1), metric_ks=[5, 10, 20, 50])
        return {"metrics": metrics, "prod_cold": prod[cold_idx], "label_cold": batch['label_id'][cold_idx]}
    
    def validation_epoch_end(self, validation_step_outputs):
        print('validation_epoch_end')
    
        prod_colds, label_colds = [], []
        for temp in validation_step_outputs:
            prod_colds.append(temp['prod_cold'])
            label_colds.append(temp['label_cold'])
        prod_colds = torch.cat(prod_colds, dim=0)
        label_colds = torch.cat(label_colds, dim=0)
        if len(label_colds) != 0:
            metrics_cold = calculate_metrics(prod_colds, label_colds.unsqueeze(1), metric_ks=[5, 10, 20, 50])
            print('tail item----------------------------------')
            for key_temp in metrics_cold:
                metrics_cold[key_temp] = round(np.mean(metrics_cold[key_temp] ) * 100, 4)
            print(metrics_cold)
            self.log("Tail Val_Metrics", metrics_cold)
            self.log("Tail Recall@10", metrics_cold['Recall@10'])
            self.logger_save.info("Tail Val Metrics: {}".format(metrics_cold))
            self.logger_save.info("Tail Recall@10: {}".format(metrics_cold['Recall@10']))
            
        # metrics_all = self.all_gather([i['metrics'] for i in validation_step_outputs])
        val_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'Recall@50': [], 'NDCG@50': []}
        val_metrics_dict_mean = {}
        for temp in validation_step_outputs:
            for key_temp, val_temp in temp['metrics'].items():
                val_metrics_dict[key_temp].append(val_temp)

        for key_temp, values_temp in val_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            val_metrics_dict_mean[key_temp] = values_mean
        print('All -----------------------------------------')
        print(val_metrics_dict_mean)
        self.log("Val_Metrics", val_metrics_dict_mean)
        self.log("Recall@10", val_metrics_dict_mean['Recall@10'])
        self.logger_save.info("Val Metrics: {}".format(val_metrics_dict_mean))
        self.logger_save.info("Recall@10: {}".format(val_metrics_dict_mean['Recall@10']))
        
    def test_step(self, batch, batch_idx):
        self.pre_imgseqrec_model.eval()
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_img = torch.stack(batch['seq_id'], dim=1)
        seq_rep_img = self.item_emb_img(seq_img)
        rep_seq = self.pre_imgseqrec_model(seq_rep_img, masks)
        prod = self.pred_layer(rep_seq)
        metrics = calculate_metrics(prod, batch['label_id'].unsqueeze(1), metric_ks=[5, 10, 20, 50])
        cold_idx = torch.where(batch['cold_item_flag']==1)
        return {"metrics": metrics, "prod_cold": prod[cold_idx], "label_cold": batch['label_id'][cold_idx]}

    def test_epoch_end(self, test_step_outputs):
        print('test_epoch_end')
        prod_colds, label_colds = [], []
        for temp in test_step_outputs:
            prod_colds.append(temp['prod_cold'])
            label_colds.append(temp['label_cold'])
        prod_colds = torch.cat(prod_colds, dim=0)
        label_colds = torch.cat(label_colds, dim=0)
        metrics_cold = calculate_metrics(prod_colds, label_colds.unsqueeze(1), metric_ks=[5, 10, 20, 50])
        print('tail item----------------------------------')
        for key_temp in metrics_cold:
            metrics_cold[key_temp] = round(np.mean(metrics_cold[key_temp] ) * 100, 4)
        print(metrics_cold)
        self.log("Tail Test_Metrics", metrics_cold)
        self.log("Tail Recall@10", metrics_cold['Recall@10'])
        self.logger_save.info("Tail Test Metrics: {}".format(metrics_cold))
        self.logger_save.info("Tail Recall@10: {}".format(metrics_cold['Recall@10']))
        
        test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'Recall@50': [], 'NDCG@50': []}
        test_metrics_dict_mean = {}
        # metrics_all = self.all_gather([i['metrics'] for i in test_step_outputs])
        for temp in test_step_outputs:
            for key_temp, val_temp in temp['metrics'].items():
                test_metrics_dict[key_temp].append(val_temp)

        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print('All -----------------------------------------')
        print(test_metrics_dict_mean)
        self.log("Test_Metrics", test_metrics_dict_mean)
        self.log("Recall@10", test_metrics_dict_mean['Recall@10'])
        self.logger_save.info("Test Metrics: {}".format(test_metrics_dict_mean))
        self.logger_save.info("Recall@10: {}".format(test_metrics_dict_mean['Recall@10']))
        

def main():
    args_parser = argparse.ArgumentParser()
    args_parser = add_optimizer_args(args_parser)
    args_parser = add_datasets_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = TailImgRec.add_module_specific_args(args_parser)
    custom_parser = [
        '--datasets_path_train', '/R2Rec/data/Beauty/imgs_seq_5_train.pickle',
        '--datasets_path_test', '/R2Rec/data/Beauty/imgs_seq_5_test.pickle',
        '--datasets_path_val', '/R2Rec/data/Beauty/imgs_seq_5_val.pickle',
        '--datasets_name', 'Beauty',
        '--train_batchsize', '128',
        '--val_batchsize', '128',
        '--test_batchsize', '128',
        '--seq_len', '10',
        '--info', 'Tail 0.2, regterm tau 0.1',
        '--learning_rate', '5e-4',
        '--min_learning_rate', '5e-5',
        '--random_seed', '512',
        '--dropout', '0.8',
        '--tau', '0.5',
        '--hidden_size', '768',
        '--attn_head', '16',
        '--n_blocks', '1',
        '--max_epochs', '100',
        '--save_ckpt_path', '/R2Rec/ckpt/temp/'
        ] 
    
    args = args_parser.parse_args(args=custom_parser)
    fix_random_seed_as(args.random_seed)
    
    if not os.path.exists('../log/'):
        os.makedirs('../log/')
    if not os.path.exists('../log/' + args.datasets_name):
        os.makedirs('../log/' + args.datasets_name)
    logging.basicConfig(level=logging.INFO, filename='../log/' + args.datasets_name + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
    logger = logging.getLogger(__name__)
    
    print(args.info)
    logger.info(args.info)
    print(args)
    logger.info(args)
    datasets = load_data(args)
    pop_items = pop_item(datasets['train'], 0.2)
    # pop_items = pop_item_interaction(datasets['train'], 0.8)
    
    collate_fn = Collator(args, pop_items)
    args.item_nums = len(collate_fn.id_asin_dict)
    
    datamodule = UniversalDataModule(collate_fn=collate_fn, args=args, datasets=datasets)
    checkpoint_callback = UniversalCheckpoint(args)
    early_stop_callback_step = EarlyStopping(monitor='Recall@10', min_delta=0.00, patience=3, verbose=False, mode='max')
    trainer = Trainer(devices=1, accelerator="gpu", strategy=DDPStrategy(find_unused_parameters=True), callbacks=[checkpoint_callback, early_stop_callback_step], max_epochs=args.max_epochs,  check_val_every_n_epoch=1)
    
    model = TailImgRec(args, collate_fn.tail_item, logger)
    ## Pretraining and Finetuning
    # model_init = torch.load('/R2Rec/ckpt/ClothingPantryMagazine_img.pt')
    # model_init = torch.load('/R2Rec/ckpt/ClothingPantryMagazine_id.pt')
    # model_init.pop('item_emb_img.weight')
    # model_init.pop('pred_layer.pred_linear.weight')
    # model.load_state_dict(model_init, strict=False)
    
    trainer.fit(model, datamodule)
    print(args)
   
    
    ## model_save
    # torch.save(model.state_dict(), '/R2Rec/ckpt/ClothingPantryMagazine_finetuning_'+args.datasets_name+'_img.pt')
    
    print('Test-------------------------------------------------------')
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
