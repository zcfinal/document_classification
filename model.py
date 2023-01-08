from transformers import BertTokenizerFast, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification



def get_model(name):
    if name=='roberta-base':
        return RobertaForSequenceClassification.from_pretrained(name)

def get_tokenizer(name):
    if name=='roberta-base':
        return RobertaTokenizer.from_pretrained(name)
