import torch
import os
from torch import nn
import logging
import numpy as np
import torch.nn.functional as F
import copy
from utils import to_cuda


def get_trainer(trainer):
    return eval(trainer)

class BaseTrainer:
    def __init__(self,args, model, train_dataloader, valid_dataloader, test_dataloader):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.best_model = None
        self.best_acc = 0

    def train(self,epoch):
        loss = []
        self.model.train()
        for step,data in enumerate(self.train_dataloader):
            data = to_cuda(data)
            rslt = self.model(**data)
            self.optimizer.zero_grad()
            rslt['loss'].backward()
            self.optimizer.step()
            loss.append(rslt['loss'].item())
            if step%self.args.logging_steps==0:
                logging.info('[Epoch {} step {}] TRAIN: loss = {:.4f}'.format(epoch+1, step, np.mean(loss)))
                loss.clear()
            if step%self.args.eval_steps==0:
                self.eval(epoch)

    def save_best_model(self,acc):
        if acc>self.best_acc:
            self.best_acc=acc
            self.best_model = copy.deepcopy(self.model.state_dict())

    def eval(self,epoch,mode='valid'):
        if mode=='valid':
            dataloader = self.valid_dataloader
        elif mode=='test':
            dataloader = self.test_dataloader
        preds = []
        labels = []
        with torch.no_grad():
            self.model.eval()
            for data in dataloader:
                data = to_cuda(data)
                rslt = self.model(**data)
                preds.append(rslt['logits'])
                labels.append(data['labels'])
            preds = torch.cat(preds,0)
            labels = torch.cat(labels,0)
            acc = (labels==(preds.argmax(-1))).float().mean().item()
            self.model.train()
        logging.info('[{}] : acc = {:.4f}'.format(mode, acc))
        if mode=='valid':
            self.save_best_model(acc)


	
    def start(self):
        self.model.to(self.args.device)
        accuracy_best = 0

        for epoch in range(self.args.num_train_epochs):
            self.train(epoch)

        self.model.load_state_dict(self.best_model)
        torch.save(self.best_model,self.args.model_dir+'/best.pt')
        self.eval(epoch,'test')


class copyTrainer(BaseTrainer):
    def __init__(self,args, model, train_dataloader, valid_dataloader, test_dataloader):
        super().__init__(args, model, train_dataloader, valid_dataloader, test_dataloader)
    
    def copy(self,epoch):
        

    def start(self):
        self.model.to(self.args.device)

        for epoch in range(self.args.num_train_epochs):
            self.copy(epoch)

        accuracy_best = 0

        for epoch in range(self.args.num_train_epochs):
            self.train(epoch)

        self.model.load_state_dict(self.best_model)
        torch.save(self.best_model,self.args.model_dir+'/best.pt')
        self.eval(epoch,'test')