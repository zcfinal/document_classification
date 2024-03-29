import os
import torch
import random
import numpy as np
import logging
from dataloader import get_dataloader
from parameters import parse_args
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from utils import set_seed, setuplogging
from utils import compute_metrics
from trainer import get_trainer


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda')
    print(torch.cuda.current_device())

    set_seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(("bert-base-uncased"))
    dataset = get_dataloader(args.dataset)(args,tokenizer)

    model = BertForSequenceClassification.from_pretrained(("bert-base-uncased"), num_labels=len(dataset.cate2id)).to(
        device)

    trainer = get_trainer(args.trainer)(args,model,dataset.train_dataloader,dataset.valid_dataloader,dataset.test_dataloader)
    trainer.start()

if __name__=='__main__':
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    setuplogging(args,0)
    logging.info(args)
    main(args)
    