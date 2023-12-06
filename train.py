PROJECT_PATH = "Structure_generation"

import sys
sys.path.append(PROJECT_PATH)
import os
import datetime
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import utils
from src import data_handler as dh
import src.GRU as gru
from src.utils import *

now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
file = os.path.basename(__file__).split(".")[0]
DIR_NAME = PROJECT_PATH + "/results/" + file + "_" + now
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
LOGGER = utils.init_logger(file,DIR_NAME,now,level_console="debug")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# argument
parser = argparse.ArgumentParser(description="CLI temprate")
parser.add_argument("--note",type=str,help="short note for this running")
parser.add_argument("--evaluate",action="store_true",default=False)
parser.add_argument("--seed",type=str,default=222)
parser.add_argument("--num_epoch",type=int,default=100)
parser.add_argument("--batch_size",type=int,default=1024)
parser.add_argument("--save_dir",type=str,default=DIR_NAME)
parser.add_argument("--patience",type=int,default=5)

args = parser.parse_args()
utils.fix_seed(seed=args.seed,fix_gpu=False)

# data preparation
def prepare_data():
    data = pd.read_csv("{}/data/2022_zinc_processed.txt".format(PROJECT_PATH),sep="\t",index_col=0)
    train_x, test_x, train_y, test_y = train_test_split(data["randomized_smiles"],data["smiles"],
                                                        test_size=0.01)
    valid_x, test_x, valid_y, test_y = train_test_split(test_x,test_y,test_size=0.5)
    return train_x, valid_x, test_x, train_y, valid_y, test_y

# model preparation
def prepare_model():
    conf = gru.Config(vocab_size=81)
    model = gru.Seq2Seq(conf).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0,reduction="mean")
    optimizer = optim.AdamW(model.parameters(),lr=0.0005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1**(1/20))
    es = gru.EarlyStopping(patience=args.patience)
    return model, criterion, optimizer, scheduler, es

def train_step(model,src,tgt,criterion,optimizer):
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)
    dec_i = tgt[:-1,:]
    target = tgt[1:,:]
    optimizer.zero_grad()
    out, _ = model(src,dec_i)
    l = criterion(out.transpose(-2,-1),target)
    l.backward()
    optimizer.step()
    return l.item()

def valid_step(model,src,tgt,criterion):
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)
    dec_i = tgt[:-1,:]
    target = tgt[1:,:]
    out, _ = model(src,dec_i)
    l = criterion(out.transpose(-2,-1),target)
    return l.item()

def train(model,train_x,train_y,valid_x,valid_y,criterion,optimizer,scheduler,es):
    model.train()
    loss = []
    valid_loss = []
    num_iter = 0
    end = False
    with tqdm(total=args.num_epoch) as pbar:
        while num_iter < args.num_epoch:
            train_loader = dh.prep_loader_sfl(train_x,train_y,buckets=(20,120,10),batch_size=args.batch_size,
                                        shuffle=True,drop_last=False,device=DEVICE)
            l = 0
            for src,tgt in train_loader:
                l += train_step(model,src,tgt,criterion,optimizer)
            loss.append(l)
            num_iter += 1

            eval_loader = dh.prep_loader_sfl(valid_x,valid_y,buckets=(20,120,10),batch_size=args.batch_size,
                                        shuffle=False,drop_last=False,device=DEVICE)
            model.eval()
            l_v = 0
            with torch.no_grad():
                for s,t in eval_loader:
                    l_v += valid_step(model,s,t,criterion)
                valid_loss.append(l_v)
            LOGGER.info(f"Step: {num_iter}, train_loss: {l:.4f}, valid_loss: {l_v:.4f}")
            if es.step(l_v):
                end = True
            model.train()

            pbar.update(1)
            if end: break
    
    return model, loss, valid_loss

def evaluate(model,x,y,maxlen=150):
    eval_loader = dh.prep_loader_sfl(x,y,buckets=(20,120,5),batch_size=args.batch_size,
                                 shuffle=False,drop_last=False,device=DEVICE)
    model.eval()
    id2sm = dh.tokens_table().id2sm
    row = []
    with torch.no_grad():
        for src, tgt in tqdm(eval_loader):
            src = src.to(DEVICE)
            dec_hid = model.encoder(src,inference=True)[1]
            token_ids = np.zeros((maxlen,src.size(1)))
            token_ids[0:,] = 1
            token_ids = torch.tensor(token_ids,device=DEVICE,dtype=torch.long)
            for i in range(1,maxlen):
                token_ids_seq = token_ids[i-1,:].unsqueeze(0)
                infer = True if i != 1 else False
                output, dec_hid = model.decode(token_ids_seq,dec_hid,inference=infer)
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

            for s,t,v in zip(src.T,tgt.T,pred.T):
                x = [id2sm[j.item()] for j in s]
                y = [id2sm[j.item()] for j in t]
                p = [id2sm[j.item()] for j in v]
                x_str = "".join(x[1:]).split(id2sm[2])[0].replace("R","Br").replace("L","Cl")
                y_str = "".join(y[1:]).split(id2sm[2])[0].replace("R","Br").replace("L","Cl")
                p_str = "".join(p).split(id2sm[2])[0].replace("R","Br").replace("L","Cl")
                judge = True if y_str == p_str else False
                row.append([x_str,y_str,p_str,judge])

        pred_df = pd.DataFrame(row,columns=["input","answer","predict","judge"])
        accuracy = len(pred_df.query("judge == True")) / len(pred_df)
    
    return pred_df, accuracy

if __name__ == "__main__":
    if args.evaluate == False:
        start = time.time()
        train_x, valid_x, test_x, train_y, valid_y, test_y = prepare_data()
        LOGGER.info(
            f"num_training_data: {len(train_x)}, num_test_data: {len(test_x)}"
        )
        model, criterion, optimizer, scheduler, es = prepare_model()
        model, loss, valid_loss = train(model,train_x,train_y,valid_x,valid_y,criterion,optimizer,scheduler,es)
        torch.save(model.state_dict(),args.save_dir+"/model.pth")
        plot_loss(args.num_epoch,loss,valid_loss,dir_name=args.save_dir)
        
        pred_df, accuracy = evaluate(model,test_x,test_y)
        pred_df.to_csv(args.save_dir+"/result.csv")
        LOGGER.info(f"accuracy: {accuracy:.4f}")

        utils.to_logger(LOGGER,name="argument",obj=args)
        utils.to_logger(LOGGER,name="loss",obj=criterion)
        utils.to_logger(LOGGER,name="optimizer",obj=optimizer,skip_keys={"state","param_groups"})
        utils.to_logger(LOGGER,name="scheduler",obj=scheduler)
        LOGGER.info("elapsed time: {:.2f} min".format((time.time()-start)/60))
    
    elif args.evaluate == True:
        start = time.time()
        train_x, valid_x, test_x, train_y, valid_y, test_y = prepare_data()
        model, criterion, optimizer, scheduler, es = prepare_model()
        model.load_state_dict(torch.load("model.pth"))
        pred_df, accuracy = evaluate(model,test_x,test_y)
        pred_df.to_csv(args.save_dir+"/result.csv")
        LOGGER.info(f"accuracy: {accuracy:.4f}")
        LOGGER.info("elapsed time: {:.2f} min".format((time.time()-start)/60))
        pass
