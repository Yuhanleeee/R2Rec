import os
import pickle
import random
import torch


def emb_merge(datalist):
    asin2id_merge_dict = {}
    count = 0
    nums  = 0
    for dataset in datalist:
        with open(os.path.join('/R2Rec/data/', dataset+'/asin2id.pickle'), 'rb') as f:
            asin2id = pickle.load(f)
            nums += len(asin2id)
            for asin in asin2id:
                if asin in asin2id_merge_dict:
                    reasin = asin + str(random.randint(0, 100))
                    while reasin in asin2id_merge_dict:
                        reasin = asin + str(random.randint(0, 100))
                    asin2id_merge_dict[reasin] = count
                else:
                    asin2id_merge_dict[asin] = count
                count += 1
    embs = []
    for dataset in datalist:
        embs.append(torch.load('/R2Rec/data/'+dataset+'/img_emb_clip_vit_large_patch14.pt'))
    embs_tensor = torch.cat(embs, dim=0)
    
    trains = []
    for dataset in datalist:
        with open(os.path.join('/R2Rec/data/'+dataset, 'imgs_seq_5_train.pickle'), 'rb') as f:
            train = pickle.load(f)
            trains += train
        
    torch.save(embs_tensor, os.path.join('/R2Rec/data/merge', ''.join(datalist)+'_img_emb_clip_vit_large_patch14.pt'))
    with open(os.path.join('/R2Rec/data/merge', ''.join(datalist)+'_asin2id.pickle'), 'wb') as f:
        pickle.dump(asin2id_merge_dict, f)
    with open(os.path.join('/R2Rec/data/merge', ''.join(datalist)+'_imgs_seq_5_train.pickle'), 'wb') as f:
        pickle.dump(trains, f)
    
    vals = []
    for dataset in datalist:
        with open(os.path.join('/R2Rec/data/'+dataset, 'imgs_seq_5_val.pickle'), 'rb') as f:
            val = pickle.load(f)
            vals += val
    tests = []
    for dataset in datalist:
        with open(os.path.join('/R2Rec/data/'+dataset, 'imgs_seq_5_test.pickle'), 'rb') as f:
            test = pickle.load(f)
            tests += test
    
    with open(os.path.join('/R2Rec/data/merge', ''.join(datalist)+'_imgs_seq_5_val.pickle'), 'wb') as f:
        pickle.dump(vals, f)
    with open(os.path.join('/R2Rec/data/merge', ''.join(datalist)+'_imgs_seq_5_test.pickle'), 'wb') as f:
        pickle.dump(tests, f)


def main():
    # merge_data = ['Beauty', 'Toys', 'Sports']
    merge_data = ['Clothing', 'Pantry', 'Magazine']
    # merge_data = ['Pantry', 'Magazine']
    emb_merge(merge_data) 


if __name__ == "__main__":
    main()