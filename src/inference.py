import pandas as pd
import torch
import torch.nn.functional as F
from data.dataset import ABSADataset
from transformers import BertModel, BertTokenizer

from models.bert_spc import BERT_SPC

MODEL_PATH = '/srv/home/ahuja/Enterpret/models/bert-spc_val_acc_0.7438.pt'


def infer(df):
    # Preprocess the dataframe
    df = ABSADataset.preprocess(df, label_present=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_SPC(bert, dropout_prob=0.1)
    model.load_state_dict(torch.load(MODEL_PATH))
    predictions = []
    model.eval()
    for i in range(len(df)):
        text, aspect = df.iloc[i]
        text_indices, segment_indices = ABSADataset.get_indices(
            text, aspect, tokenizer)
        try:
            output = model([text_indices, segment_indices])
        except Exception as e:
            print(e, text, aspect)
            continue
        probs = F.softmax(output, dim=-1).detach().numpy()
        pred = probs.argmax(axis=-1) - 1
        predictions.append(pred)
    df['labels'] = predictions
    return df


if __name__ == "__main__":
    df = pd.read_csv("/srv/home/ahuja/Enterpret/data/test.csv")
    df_fin = infer(df)
    df_fin.to_csv('ans.csv', index=False)
