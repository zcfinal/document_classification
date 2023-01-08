from torch.utils.data import DataLoader, Dataset
import tqdm
import torch
import logging
import jsonlines

def get_dataloader(dataloader):
    return eval(dataloader)

class NewsDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
    
    def __getitem__(self,idx):
        data = {}
        for key in self.data:
            data[key]=self.data[key][idx]
        return data

    def __len__(self):
        return len(self.data['labels'])

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        print('Initailize Dataset...')
        self.dataset = []
        self.label = []
        self.tokenizer = tokenizer
        self.args = args
        with jsonlines.open('../../data/data_0107.jsonl','r') as f:
            for line in f:
                Q = line["Q"]
                A_human = line["A_human"]
                A_chatgpt = line["A_chatgpt"]
                pos = Q + ' ' + A_human
                neg = Q + ' ' + A_chatgpt
                self.dataset.append(pos)
                self.label.append(1)
                self.dataset.append(neg)
                self.label.append(0)

        self.dataset = self.tokenizer(self.dataset, return_tensors='pt', padding='max_length', max_length=args.max_length, truncation=True)
        self.dataset['labels'] = self.label
        len_data = len(self.label)
        train_data = {}
        valid_data = {}
        test_data = {}
        for key in self.dataset:
            train_data[key] = self.dataset[key][:int(0.8*len_data)]
            valid_data[key] = self.dataset[key][int(0.8*len_data):int(0.9*len_data)]
            test_data[key] = self.dataset[key][int(0.9*len_data):]

        train_dataset = NewsDataset(train_data)
        logging.info(f'traindata len:{len(train_dataset)}')
        valid_dataset = NewsDataset(valid_data)
        logging.info(f'validdata len:{len(valid_dataset)}')
        test_dataset = NewsDataset(test_data)
        logging.info(f'testdata len:{len(test_dataset)}')

        def collate_fn(data):
            data_batch = {key:[] for key in list(data[0].keys())}
            for x in data:
                for key in x:
                    data_batch[key].append(x[key])
            for key in data_batch:
                if key=='labels':
                    data_batch[key] = torch.LongTensor(data_batch[key])
                else:
                    data_batch[key] = torch.stack(data_batch[key],0)
            return data_batch

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size,shuffle=True,collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.per_device_eval_batch_size,collate_fn=collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.per_device_eval_batch_size,collate_fn=collate_fn)


class Kfold_GPTDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        print('Initailize Dataset...')
        self.dataset = []
        self.label = []
        self.tokenizer = tokenizer
        self.args = args
        self.kfold=args.kfold
        with jsonlines.open('../../data/data_0107.jsonl','r') as f:
            for line in f:
                Q = line["Q"]
                A_human = line["A_human"]
                A_chatgpt = line["A_chatgpt"]
                pos = Q + ' ' + A_human
                neg = Q + ' ' + A_chatgpt
                self.dataset.append(pos)
                self.label.append(1)
                self.dataset.append(neg)
                self.label.append(0)

        self.dataset = self.tokenizer(self.dataset, return_tensors='pt', padding='max_length', max_length=args.max_length, truncation=True)
        self.dataset['labels'] = torch.tensor(self.label)
        
    def get_kfold(self,k):
        len_data = len(self.label)
        fold_size = len_data//self.kfold
        train_data = {}
        test_data = {}

        for key in self.dataset:
            train_data[key] = torch.cat([self.dataset[key][:k*fold_size], self.dataset[key][(k+1)*fold_size:]],0)
            test_data[key] = self.dataset[key][k*fold_size:(k+1)*fold_size]

        train_dataset = NewsDataset(train_data)
        logging.info(f'traindata len:{len(train_dataset)}')
        test_dataset = NewsDataset(test_data)
        logging.info(f'testdata len:{len(test_dataset)}')

        def collate_fn(data):
            data_batch = {key:[] for key in list(data[0].keys())}
            for x in data:
                for key in x:
                    data_batch[key].append(x[key])
            for key in data_batch:
                if key=='labels':
                    data_batch[key] = torch.LongTensor(data_batch[key])
                else:
                    data_batch[key] = torch.stack(data_batch[key],0)
            return data_batch

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size,shuffle=True,collate_fn=collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.per_device_eval_batch_size,collate_fn=collate_fn)

class Kfold_TFIDFDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        print('Initailize Dataset...')
        self.dataset = {'input':[]}
        self.label = []
        self.tokenizer = tokenizer
        self.args = args
        self.kfold=args.kfold
        with jsonlines.open('../../data/data_0107.jsonl','r') as f:
            for line in f:
                Q = line["Q"]
                A_human = line["A_human"]
                A_chatgpt = line["A_chatgpt"]
                pos = Q + ' ' + A_human
                neg = Q + ' ' + A_chatgpt
                self.dataset['input'].append(pos)
                self.label.append(1)
                self.dataset['input'].append(neg)
                self.label.append(0)

        self.dataset['labels'] = self.label
        
    def get_kfold(self,k):
        len_data = len(self.label)
        fold_size = len_data//self.kfold
        train_data = {}
        test_data = {}

        for key in self.dataset:
            train_data[key] = self.dataset[key][:k*fold_size]+ self.dataset[key][(k+1)*fold_size:]
            test_data[key] = self.dataset[key][k*fold_size:(k+1)*fold_size]

        train_data['input'] = self.tokenizer.fit_transform(train_data['input'])
        logging.info(f'traindata len:{len(train_data["labels"])}')
        test_data['input'] = self.tokenizer.transform(test_data['input'])
        logging.info(f'testdata len:{len(test_data["labels"])}')

        self.train_dataloader = train_data
        self.test_dataloader = test_data


class MINDDataset:
    def __init__(self,args,tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.news2idx = {}
        self.data = []
        self.label = []
        self.cate2id = {}
        self.subcate2id = {}
        self.file_list = ['MINDlarge_test', 'MINDlarge_train', 'MINDlarge_dev']
        self.read_data()


    def read_data(self):
        for f in tqdm.tqdm(self.file_list, desc="读取new.tsv"):
            for line in open(f'/home/v-chaozhang/mind_data/MIND/{f}/news.tsv', encoding='utf-8'):
                news_id, category, subcategory, title, abstract, url, title_entities, abs_entites = line.strip().split('\t')
                if category not in self.cate2id.keys():
                    self.cate2id[category] = len(self.cate2id)
                if subcategory not in self.subcate2id.keys():
                    self.subcate2id[subcategory] = len(self.subcate2id)
                if news_id not in self.news2idx:
                    self.news2idx[news_id] = len(self.news2idx)
                    self.data.append(title)
                    self.label.append(self.cate2id[category])
        
        tokenized_title = self.tokenizer(self.data, truncation=True, padding=True, max_length=self.args.max_length)
        self.data = tokenized_title   
        self.data['labels'] = self.label
        len_data = len(self.label)
        train_data = {}
        valid_data = {}
        test_data = {}
        for key in self.data:
            train_data[key] = self.data[key][:int(0.8*len_data)]
            valid_data[key] = self.data[key][int(0.8*len_data):int(0.9*len_data)]
            test_data[key] = self.data[key][int(0.9*len_data):]

        train_dataset = NewsDataset(train_data)
        logging.info(f'traindata len:{len(train_dataset)}')
        valid_dataset = NewsDataset(valid_data)
        logging.info(f'validdata len:{len(valid_dataset)}')
        test_dataset = NewsDataset(test_data)
        logging.info(f'testdata len:{len(test_dataset)}')

        def collate_fn(data):
            data_batch = {key:[] for key in list(data[0].keys())}
            for x in data:
                for key in x:
                    data_batch[key].append(x[key])
            for key in data_batch:
                data_batch[key] = torch.LongTensor(data_batch[key])
            return data_batch

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size,shuffle=True,collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.per_device_eval_batch_size,collate_fn=collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.per_device_eval_batch_size,collate_fn=collate_fn)

