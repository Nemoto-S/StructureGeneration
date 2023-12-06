import numpy as np
import random
import torch
import logging
import datetime
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


class EarlyStopping():
    def __init__(self,mode="min",min_delta=0,patience=10,percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode,min_delta,percentage)

        if patience == 0:
            self.is_better = lambda a,b: True
            self.step = lambda a: False

    def step(self,metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics,self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print("terminating because of early stopping.")
            return True
        
        return False

    def _init_is_better(self,mode,min_delta,percentage):
        if mode not in {"min","max"}:
            raise ValueError("mode "+mode+" is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best*min_delta/100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best*min_delta/100)


def fix_seed(seed=None,fix_gpu=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_logger(module_name,outdir="",tag="",level_console="warning",level_file="info"):
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag) == 0:
        tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(
        level = level_dic[level_file],
        filename = f"{outdir}/log_{tag}.txt",
        format = '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt = '%Y%m%d-%H%M%S'
    )
    logger = logging.getLogger(module_name)
    sh = TqdmLoggingHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
    )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def to_logger(logger,name="",obj=None,skip_keys=set(),skip_hidden=True):
    logger.info(name)
    for k,v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden:
                if not k.startswith("_"):
                    logger.info("   {0}: {1}".format(k,v))
            else:
                logger.info("   {0}: {1}".format(k,v))


class TqdmLoggingHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            tqdm.write(msg,file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


def plot_loss(train_loss_list,valid_loss_list=[],dir_name=""):
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(111)
    plt.rcParams["font.size"] = 18
    ax1.plot(train_loss_list,color="blue",label="train")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.grid()

    if len(valid_loss_list) > 0:
        n = len(train_loss_list) // len(valid_loss_list)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(n,len(train_loss_list)+1,n),valid_loss_list,color="orange",label="valid")
        ax2.grid()

    if len(dir_name) > 0:
        plt.savefig(dir_name+"/loss.png",bbox_inches="tight")
    else:
        plt.show()