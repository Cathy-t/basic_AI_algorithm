import json
import numpy as np
import os 

import torch 
from torch.utils.data import Dataset, DataLoader
import random
from utils import padding_idx

def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')  # encoding='latin1' to handle the inconsistency between python 2 and 3
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))
        feats = feats[inds]
    assert feats.shape[0] == max_length
    return np.float32(feats)

class vatex_dataset(Dataset):
    def __init__(self, data_dir, file_path, img_dir, split_type, tokenizers, max_vid_len, pair):
        src, tgt = pair
        maps = {'en':'enCap', 'zh':'chCap'}
        self.data_dir = data_dir
        self.img_dir = img_dir
        # load tokenizer
        self.tok_src, self.tok_tgt = tokenizers
        self.max_vid_len = max_vid_len
        self.split_type = split_type

        with open(self.data_dir+file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.srccaps, self.tgtcaps = [], []
        self.sent_ids = []
        for d in data[:3]:
            srccap = d[maps[src]]
            self.srccaps.extend(srccap)
            sent_id = [''.join((d['videoID'], '&', str(i))) for i in range(len(srccap))]
            self.sent_ids.extend(sent_id)
            tgtcap = d[maps[tgt]]
            self.tgtcaps.extend(tgtcap)

    def __len__(self):
        return len(self.srccaps)

    def __getitem__(self, idx):
        str_srccap,  sent_id = self.srccaps[idx], self.sent_ids[idx]
        srccap, srccap_mask, caplen_src = self.tok_src.encode_sentence(str_srccap)
        srcref = self.tok_src.encode_sentence_nopad_2str(str_srccap)

        if self.split_type != 'test':
            str_tgtcap = self.tgtcaps[idx]
            tgtcap, tgt_mask, caplen_tgt = self.tok_tgt.encode_sentence(str_tgtcap, flag='TRG')
            tgtref = self.tok_tgt.encode_sentence_nopad_2str(str_tgtcap)

            return srccap, srccap_mask, tgtcap, tgt_mask, caplen_tgt, srcref, tgtref
        else:
            str_tgtcap = self.tgtcaps[idx]
            tgtcap, _, _ = self.tok_tgt.encode_sentence(str_tgtcap, flag='TRG')
            tgtref = self.tok_tgt.encode_sentence_nopad_2str(str_tgtcap)
            return srccap, srccap_mask, sent_id, tgtcap[1:], srcref, tgtref


def get_loader(data_dir, tokenizers, split_type, batch_size, max_vid_len, pair, num_workers, pin_memory):
    maps = {'train':['vatex_train_data.json', 'trainval'], 'val': ['vatex_valid_data.json', 'trainval'],
        'test': ['vatex_test_data.json', 'trainval']}
    file_path, img_dir = maps[split_type]
    mydata = vatex_dataset(data_dir, file_path, img_dir, split_type, tokenizers, max_vid_len, pair)
    if split_type in ['train']:
        shuffle = True
    elif split_type in ['val', 'test']:
        shuffle = False
    myloader = DataLoader(dataset=mydata, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return myloader


def create_split_loaders(data_dir, tokenizers, batch_size, max_vid_len, pair, num_workers=0, pin_memory=False):
    train_loader = get_loader(data_dir, tokenizers, 'train', batch_size, max_vid_len, pair, num_workers, pin_memory)
    val_loader = get_loader(data_dir, tokenizers, 'val', batch_size, max_vid_len, pair, num_workers, pin_memory)
    test_loader = get_loader(data_dir, tokenizers, 'test', 1, max_vid_len, pair, num_workers, pin_memory)
    # test_loader = [0]

    return train_loader, val_loader, test_loader

