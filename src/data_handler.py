import numpy as np
import pandas as pd
import re
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

CHARSET = {"smiles":r"Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps\/\\]"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BucketSampler(Sampler):
    def __init__(self,dataset,buckets=(20,110,10),shuffle=True,batch_size=128,drop_last=False,drop_right=True,device="cuda"):
        super().__init__(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        length = [len(v[0]) for v in dataset]
        bucket_range = np.arange(*buckets)
        
        assert isinstance(buckets,tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0
        buc = torch.bucketize(torch.tensor(length),torch.tensor(bucket_range),right=False)

        bucs = defaultdict(list)
        bucket_max = max(np.array(buc))
        for i,v in enumerate(buc):
            bucs[v.item()].append(i)
        if (drop_right == True) and (bucket_max - 1 == (bmax - bmin) // bstep):
            _ = bucs.pop(bucket_max)
        
        self.buckets = dict()
        for bucket_size, bucket in bucs.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket,dtype=torch.int,device=device)
        self.__iter__()

    def __iter__(self):
        for bucket_size in self.buckets.keys():
            self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket,self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)
        if self.shuffle == True:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length
    

class Seq2id_Dataset(Dataset):
    def __init__(self,x,y):
        self.input = seq2id(x,vocab_dict())
        self.output = seq2id(y,vocab_dict())
        self.datanum = len(x)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_i = self.input[idx]
        out_o = self.output[idx]
        return out_i, out_o
    

class SFL_Dataset(Seq2id_Dataset):
    def __init__(self,x,y):
        self.tokens = tokens_table()
        self.input = sfl_seq2id(x,self.tokens)
        self.output = sfl_seq2id(y,self.tokens)
        self.datanum = len(x)


class tokens_table():
    def __init__(self):
        tokens = ['<pad>','<s>','</s>','0','1','2','3','4','5','6','7','8','9','(',')','=','#','@','*','%',
                  '.','/','\\','+','-','c','n','o','s','p','H','B','C','N','O','P','S','F','L','R','I',
                  '[C@H]','[C@@H]','[C@@]','[C@]','[CH2-]','[CH-]','[C+]','[C-]','[CH]','[C]','[H+]','[H]',
                  '[n+]','[nH]','[N+]','[NH+]','[NH-]','[N+]','[N@]','[N@@]','[NH2+]','[N-]','[N]''[NH]',
                  '[O+]','[O-]','[OH-]','[O]','[S]','[S+]','[s+]','[S@]','[S@@]','[B-]','[P]','[P+]','[P@]','[P@@]',
                  '[Cl]','[Cl-]','[I-]','[Br-]',"[Si]"]
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.table_len = len(self.table)

def collate(batch):
    xs, ys = [], []
    for x,y in batch:
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    ys = pad_sequence(ys,batch_first=False,padding_value=0)
    return xs, ys

def prep_loader(data_x,data_y,buckets=(20,110,10),batch_size=128,shuffle=True,drop_last=False,device=DEVICE):
    datasets = Seq2id_Dataset(data_x,data_y)
    bucket_sampler = BucketSampler(datasets,buckets=buckets,shuffle=shuffle,batch_size=batch_size,
                                   drop_last=drop_last,device=device)
    train_loader = DataLoader(datasets,batch_sampler=bucket_sampler,collate_fn=collate)                                    
    return train_loader

def prep_loader_sfl(data_x,data_y,buckets=(20,110,10),batch_size=128,shuffle=True,drop_last=False,device=DEVICE):
    datasets = SFL_Dataset(data_x,data_y)
    bucket_sampler = BucketSampler(datasets,buckets=buckets,shuffle=shuffle,batch_size=batch_size,
                                   drop_last=drop_last,device=device)
    train_loader = DataLoader(datasets,batch_sampler=bucket_sampler,collate_fn=collate)
    return train_loader

def vocab_dict(charset="smiles"):
    regex_sml = CHARSET[charset]
    a = r"Cl|Br|#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps\/\\"
    temp = re.findall(regex_sml,a)
    temp = sorted(set(temp),key=temp.index)
    vocab_smi = {}
    for i,v in enumerate(temp):
        vocab_smi[v] = i+3
    vocab_smi.update({"<pad>":0,"<s>":1,"</s>":2}) 
    return vocab_smi

def seq2id(seq_list,vocab,charset="smiles"):
    regex_sml = CHARSET[charset]
    idx_list = []
    ap = idx_list.append
    for v in seq_list:
        char = re.findall(regex_sml,v)
        seq = np.array([vocab[w] for w in char])
        seq = np.concatenate([np.array([1]),seq,np.array([2])]).astype(np.int32)
        idx_list.append(seq)
    return idx_list

def sfl_seq2id(smiles,tokens):
    tokenized = sfl_tokenize(smiles,tokens.table)
    encoded = one_hot_encoder(tokenized,tokens.dict)
    return encoded

def sfl_tokenize(smiles,token_list):
    tokenized = []
    for smile in smiles:
        smile = smile.replace("Br","R").replace("Cl","L")
        char = ""
        tok = []
        for s in smile:
            char += s
            if char in token_list:
                tok.append(char)
                char = ""
        tokenized.append(tok)
    return tokenized

def one_hot_encoder(tokenized,token_dict):
    encoded = []
    for token in tokenized:
        enc = np.array([token_dict[v] for v in token])
        enc = np.concatenate([np.array([1]),enc,np.array([2])]).astype(np.int32)
        encoded.append(enc)
    return encoded
