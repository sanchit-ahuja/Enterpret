import pandas as pd
import torch
import torch.nn.functional as F
from data.dataset import ABSADataset
from transformers import BertModel, BertTokenizer
import os
from models.bert_spc import BERT_SPC

MODEL_PATH = os.path.abspath("models/bert-spc_val_acc_0.7738.pt")
DATA_PATH = os.path.abspath("data/test.csv")
SAVE_PATH = os.path.abspath("results/test.csv")

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
        output = model([text_indices, segment_indices])
        probs = F.softmax(output, dim=-1).detach().numpy()
        pred = probs.argmax(axis=-1) - 1
        predictions.append(pred[0])
    df['labels'] = predictions
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df_fin = infer(df)
    df_fin.to_csv(SAVE_PATH, index=False)
