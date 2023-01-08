import os
import torch
import random
import numpy as np
import logging
from model import get_model, get_tokenizer
from dataloader import get_dataloader
from parameters import parse_args
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import Trainer, TrainingArguments
from utils import set_seed, setuplogging
from utils import compute_metrics
from trainer import get_trainer


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda')
    print(torch.cuda.current_device())

    set_seed(args.seed)

    tokenizer =get_tokenizer(args.tokenizer_name)
    dataset = get_dataloader(args.dataset)(args,tokenizer)

    model = get_model(args.model_name).to(device)
    if args.load_model_path is not None:
        print('load model')
        state = torch.load(args.load_model_path)['model_state_dict']
        model.load_state_dict(state,strict=False)

    trainer = get_trainer(args.trainer)(args,model,dataset)
    trainer.start()

if __name__=='__main__':
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    setuplogging(args,0)
    logging.info(args)
    main(args)
    