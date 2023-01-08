import argparse
from operator import truediv
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument('--load_model_path',type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--trainer", type=str, default='BaseTrainer')
    parser.add_argument("--dataset", type=str, default='GPTDataset')
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--kfold",type=int,default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #args = parse_args()
    import torch
    import numpy as np
    log_mask=torch.tensor([1,1,1,1,0,0,0]).cuda()
    print(log_mask.device)
    a=np.random.uniform(0,1,size=tuple(log_mask.size()))
    c=np.random.uniform(0,1)
    b=np.array(a>c)
    print(a)
    print(c)
    print(torch.FloatTensor(b).to(log_mask.device))

