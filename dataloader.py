from torch.utils.data import DataLoader, Dataset
import tqdm
import torch
import logging

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

