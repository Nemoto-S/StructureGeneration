import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm.auto import tqdm
from rdkit import RDLogger, Chem
from itertools import chain
import torch
import torch.nn as nn

import src.data_handler as dh
import src.GRU as gru

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RDLogger.DisableLog("rdApp.*")

conf = gru.Config(vocab_size=81)
model = gru.Seq2Seq(conf).to(DEVICE)
model.load_state_dict(torch.load("model/GRU_SFL_96model.pth"))

count = 0
generated = []
with torch.no_grad():
    model.eval()
    while len(generated) < 1000000:
        latent = torch.rand(1024,256) * 2 - 1
        latent = latent.to(DEVICE)
        token_ids = torch.zeros(150,latent.size(0),dtype=torch.long).to(DEVICE)
        token_ids[0:,] = 1
        for i in range(1,150):
            token_ids_seq = token_ids[i-1,:].unsqueeze(0)
            infer = True if i != 1 else False
            output, latent = model.decode(token_ids_seq,latent,inference=infer)
            _, new_id = output.max(dim=2)
            is_end_token = token_ids_seq == 2
            is_pad_token = token_ids_seq == 0
            judge = torch.logical_or(is_end_token,is_pad_token)
            if judge.sum().item() == judge.numel():
                token_ids = token_ids[:i,:]
                break
            new_id[judge] = 0
            token_ids[i,:] = new_id
        pred = token_ids[1:,:]

        g = []
        for v in pred.T:    
            id2sm = dh.tokens_table().id2sm
            p = [id2sm[j.item()] for j in v]
            p_str = "".join(p).split(id2sm[2])[0].replace("R","Br").replace("L","Cl")
            g.append(p_str)

        for v in g:
            try:
                m = Chem.MolFromSmiles(v)
                s = Chem.MolToSmiles(m)
                generated.append(s)
            except:
                pass
        count += 1024
    
generated = list(set(generated))

with torch.no_grad():
    model.eval()
    while len(generated) < 1000000:
        latent = torch.rand(1024,256) * 2 - 1
        latent = latent.to(DEVICE)
        token_ids = torch.zeros(150,latent.size(0),dtype=torch.long).to(DEVICE)
        token_ids[0:,] = 1
        for i in range(1,150):
            token_ids_seq = token_ids[i-1,:].unsqueeze(0)
            infer = True if i != 1 else False
            output, latent = model.decode(token_ids_seq,latent,inference=infer)
            _, new_id = output.max(dim=2)
            is_end_token = token_ids_seq == 2
            is_pad_token = token_ids_seq == 0
            judge = torch.logical_or(is_end_token,is_pad_token)
            if judge.sum().item() == judge.numel():
                token_ids = token_ids[:i,:]
                break
            new_id[judge] = 0
            token_ids[i,:] = new_id
        pred = token_ids[1:,:]

        g = []
        for v in pred.T:    
            id2sm = dh.tokens_table().id2sm
            p = [id2sm[j.item()] for j in v]
            p_str = "".join(p).split(id2sm[2])[0].replace("R","Br").replace("L","Cl")
            g.append(p_str)

        for v in g:
            try:
                m = Chem.MolFromSmiles(v)
                s = Chem.MolToSmiles(m)
                generated.append(s)
            except:
                pass
        count += 1024
   
with open("generated_smiles_1m.txt","w") as f:
    f.write("\n".join(generated))
