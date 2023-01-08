import torch
import os
from torch import nn
import logging
import numpy as np
import torch.nn.functional as F
import copy
from utils import to_cuda
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

def get_trainer(trainer):
    return eval(trainer)

class BaseTrainer:
    def __init__(self,args, model, dataset):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.train_dataloader = dataset.train_dataloader
        self.valid_dataloader = dataset.valid_dataloader
        self.test_dataloader = dataset.test_dataloader
        self.best_model = None
        self.best_acc = 0

    def train(self,epoch):
        loss = []
        self.model.train()
        for step,data in enumerate(self.train_dataloader,start=1):
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
        epoch=0
        self.eval(epoch,'test')
        for epoch in range(self.args.num_train_epochs):
            self.train(epoch)

        self.model.load_state_dict(self.best_model)
        torch.save(self.best_model,self.args.model_dir+'/best.pt')
        self.eval(epoch,'test')


class KfoldTrainer(BaseTrainer):
    def __init__(self,args, model, dataset):
        self.args = args
        self.model = model
        self.init_state = copy.deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.dataset = dataset
        self.best_model = None
        self.best_acc = 0

    def eval(self,epoch,mode='test'):
        if mode=='test':
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
        self.save_best_model(acc)

    def start(self):
        self.model.to(self.args.device)
        best_result = []
        zero_result = []
        for k in range(self.args.kfold):
            self.model.load_state_dict(self.init_state)
            logging.info('[{}-fold] training begin.'.format(k))
            self.dataset.get_kfold(k)
            self.train_dataloader = self.dataset.train_dataloader
            self.test_dataloader = self.dataset.test_dataloader
            self.best_acc = 0
            epoch=0
            self.eval(epoch,'test')
            zero_result.append(self.best_acc)

            for epoch in range(self.args.num_train_epochs):
                self.train(epoch)

            self.model.load_state_dict(self.best_model)
            torch.save(self.best_model,self.args.model_dir+f'/{k}-fold-best.pt')
            self.eval(epoch,'test')
            best_result.append(self.best_acc)
        logging.info(f'{self.args.kfold}-fold best score: {np.mean(best_result)}')
        logging.info(f'{self.args.kfold}-fold zero shot score: {np.mean(zero_result)}')

class KfoldTFIDFTrainer(BaseTrainer):
    def __init__(self,args, model, dataset):
        self.args = args
        self.model = model
        self.init_state = copy.deepcopy(self.model.get_params())
        self.dataset = dataset
        self.best_model = None
        self.best_acc = 0

    def eval(self,epoch,mode='test'):
        if mode=='test':
            dataloader = self.test_dataloader
        acc = self.model.score(self.test_dataloader['input'], self.test_dataloader['labels'])
        logging.info('[{}] : acc = {:.4f}'.format(mode, acc))
        self.save_best_model(acc)

    def save_best_model(self,acc):
        if acc>self.best_acc:
            self.best_acc=acc
            self.best_model = copy.deepcopy(self.model.get_params())

    def train(self,epoch):
        self.model.fit(self.train_dataloader['input'], self.train_dataloader['labels'])

    def start(self):
        best_result = []
        zero_result = []
        for k in range(self.args.kfold):
            logging.info('[{}-fold] training begin.'.format(k))
            self.dataset.get_kfold(k)
            self.train_dataloader = self.dataset.train_dataloader
            self.test_dataloader = self.dataset.test_dataloader
            params = {'C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
            split = PredefinedSplit([-1]*len(self.train_dataloader['labels'])+[0]*len(self.test_dataloader['labels']))
            search = GridSearchCV(self.model, params, cv=split, n_jobs=None, verbose=True, refit=False)
            search.fit(sparse.vstack([self.train_dataloader['input'], self.test_dataloader['input']]), self.train_dataloader['labels']+self.test_dataloader['labels'])
            self.model = self.model.set_params(**search.best_params_)
            self.best_acc = 0
            epoch=0
            self.train(epoch)
            self.eval(epoch,'test')
            best_result.append(self.best_acc)
        logging.info(f'{self.args.kfold}-fold best score: {np.mean(best_result)}')
        logging.info(f'{self.args.kfold}-fold zero shot score: {np.mean(zero_result)}')