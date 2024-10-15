import os
import pickle
import json
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
import torch
import numpy as np

dataset_name_dict = {'Sports': 'Sports_and_Outdoors_5',
                     'Pantry': 'Prime_Pantry',
                     'Magazine': 'Magazine_Subscriptions'}


def title_description_save(path_data, dataset, savepathdir):
    files = os.listdir(os.path.join(path_data, 'img'))
    list_asin = []
    for file in files:
        if '.jpg' in file:
            list_asin.append(file.split('.jpg')[0])
    asin_title_description_dict = {}
    with open(os.path.join(path_data, 'meta_'+ dataset_name_dict[dataset] +'.json'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if 'title' in data:
                title = data['title']
            else:
                title = None
            if 'description' in data and len(data['description']) > 0:
                description = data['description'][0]
            else:
                description = None
            asin_title_description_dict[data['asin']] = [title, description]
    asin_img_tilte_description_dict = {}
    
    for asin in list_asin:
        asin_img_tilte_description_dict[asin] = asin_title_description_dict[asin]
    
    if not os.path.exists(savepathdir+dataset):  
        os.makedirs(savepathdir+dataset)

    with open(os.path.join(savepathdir+dataset, 'imgs_asin_title_description.pickle'), 'wb') as f:
        pickle.dump(asin_img_tilte_description_dict, f)


def seq_filter_5_core(path_data, dataset_name, savedata_dir):
    with open(os.path.join(savedata_dir+dataset_name, 'imgs_asin_title_description.pickle'), 'rb') as f:
        data_asin_title_description = pickle.load(f)
    asins = list(data_asin_title_description.keys())
    reviewer_item_time_dict = {}
    with open(os.path.join(path_data+dataset_name, dataset_name_dict[dataset_name]+'.json'), 'r') as f:
        for data in f:
            line = json.loads(data)
            if 'asin' in line and 'reviewerID' in line and 'unixReviewTime' in line:
                asin = line['asin']
                if asin in asins:
                    reviewerid = line['reviewerID']
                    timestamp = line['unixReviewTime']
                    if reviewerid not in reviewer_item_time_dict:
                        reviewer_item_time_dict[reviewerid] = [(asin, timestamp)]
                    else:
                        reviewer_item_time_dict[reviewerid].append((asin, timestamp))
         
    filter_len_5_reviewer_item_time_dict = {}
    for reviewerid_temp in reviewer_item_time_dict:
        if len(reviewer_item_time_dict[reviewerid_temp]) >= 5:
            seq_temp, time_seq_temp = [], []
            for temp in reviewer_item_time_dict[reviewerid_temp]: 
                seq_temp.append(temp[0])
                time_seq_temp.append(temp[1])
            id_temp = sorted(range(len(time_seq_temp)), key=lambda k: time_seq_temp[k])
            filter_len_5_reviewer_item_time_dict[reviewerid_temp] = [seq_temp[i] for i in id_temp]
    
    with open(os.path.join(savedata_dir+dataset_name, 'imgs_seq_5.pickle'), 'wb') as f:
        pickle.dump(filter_len_5_reviewer_item_time_dict, f)


def seq_filter_5_core_split(path_data):
    path_data_seq = os.path.join(path_data, 'imgs_seq_5.pickle')
    data_train = []
    data_val = []
    data_test = []
    with open(path_data_seq, 'rb') as f:
        data_seq = pickle.load(f)
    for user_temp in data_seq:
        data_train.append(data_seq[user_temp][:-2])
        data_val.append(data_seq[user_temp][:-1])
        data_test.append(data_seq[user_temp])
    with open(os.path.join(path_data, 'imgs_seq_5_train.pickle'), 'wb') as f:
        pickle.dump(data_train, f)
    with open(os.path.join(path_data, 'imgs_seq_5_val.pickle'), 'wb') as f:
        pickle.dump(data_val, f)
    with open(os.path.join(path_data, 'imgs_seq_5_test.pickle'), 'wb') as f:
        pickle.dump(data_test, f)


def asin2id_save(path_data):
    with open(os.path.join(path_data, 'imgs_seq_5.pickle'), 'rb') as f:
        data_seq = pickle.load(f)
    seq_list = []
    for temp in data_seq:
        seq_list += data_seq[temp]
    asin_id_dict = {}
    count = 0
    for asin_temp in seq_list:
        if asin_temp not in asin_id_dict:
            asin_id_dict[asin_temp] = count
            count += 1
    with open(os.path.join(path_data, 'asin2id.pickle'), 'wb') as f:
        pickle.dump(asin_id_dict, f)


def img_emb_vit(path_data, model_path, img_path):
    with open(os.path.join(path_data, 'asin2id.pickle'), 'rb') as f:
        data_asin_id_dict = pickle.load(f)
    image_processor = ViTImageProcessor.from_pretrained(model_path)
    vit_model = ViTModel.from_pretrained(model_path)
    img_load_path = img_path + '/img/'
    list_embs_array = []
    for img_temp in data_asin_id_dict:
        path_img = img_load_path + img_temp + '.jpg'
        try:
            img = Image.open(path_img)
        except:
            img = Image.open('/R2Rec/data/error.jpg')
        try:
            img_tensor = image_processor(img, return_tensors="pt")['pixel_values']
        except:
            img_tensor = image_processor(img.convert('RGB'), return_tensors="pt")['pixel_values']
        emb_temp = vit_model(img_tensor).last_hidden_state.squeeze(0)[0, :].detach().numpy()
        list_embs_array.append(emb_temp)
    embs_tensor = torch.from_numpy(np.stack(list_embs_array, axis=0))
    torch.save(embs_tensor, os.path.join(path_data, 'img_emb_clip_vit_large_patch14.pt'))


def main():
    # dataname_pretrain = ['Magazine', 'Clothing', 'Pantry']
    dataname_pretrain = ['Pantry', 'Magazine']
    pathdir = '/R2Rec/data/'
    savepathdir = '/R2Rec/data/'
    imgmodel_path = '/R2Rec/model_load/clip-vit-large-patch14'
    for dataname in dataname_pretrain:
        # title_description_save(pathdir+dataname, dataname, savepathdir)
        # seq_filter_5_core(pathdir, dataname, savepathdir)
        # seq_filter_5_core_split(savepathdir+dataname)
        # asin2id_save(savepathdir+dataname)
        img_emb_vit(savepathdir+dataname, imgmodel_path, pathdir+dataname)
        
        
if __name__ == "__main__":
    main()