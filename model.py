from transformers import BertTokenizerFast, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def get_model(name):
    if name=='roberta-base':
        return RobertaForSequenceClassification.from_pretrained(name)
    elif name=='logistic':
        model = LogisticRegression(solver='liblinear')
        return model

def get_tokenizer(name):
    if name=='roberta-base':
        return RobertaTokenizer.from_pretrained(name)
    elif name=='tf-idf':
        return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)

