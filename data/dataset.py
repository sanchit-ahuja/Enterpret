import re
import string

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class ABSADataset(Dataset):

    @staticmethod
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    @staticmethod
    def preprocess(df, label_present=True):
        df['text'] = df['text'].apply(lambda x: ' '.join(x.split()))
        df['text'] = df['text'].apply(lambda x: str(x).lower())
        if label_present:
            df['label'] = df['label'].apply(lambda x: int(x))
        df['aspect'] = df['aspect'].apply(lambda x: str(x).lower())
        df['text'] = df['text'].apply(lambda x: ABSADataset.remove_emoji(x))
        df['aspect'] = df['aspect'].apply(
            lambda x: ABSADataset.remove_emoji(x))
        df['text'] = df['text'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.punctuation)))  # removing punctuation
        df['aspect'] = df['aspect'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.punctuation)))  # removing punctuation
        df['text'] = df['text'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.digits)))  # removing integers
        df['aspect'] = df['aspect'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.digits)))  # removing integers
        return df

    @staticmethod
    def get_indices(text, aspect, tokenizer):
        text_len = len(text.split())
        aspect_len = len(aspect.split())
        text_indices = tokenizer(
            str(text),
            str(aspect),
            padding='max_length',
            max_length=512,  # Max length assuming
            return_tensors='pt',
            add_special_tokens=True,
        )

        segment_indices = ([0] * (text_len + 2) + [1]*(aspect_len + 1))
        segment_indices = torch.LongTensor(
            pad_and_truncate(segment_indices, 512))
        return text_indices['input_ids'].squeeze(0), segment_indices

    def __init__(self, df, tokenizer):
        self.df = ABSADataset.preprocess(df)
        self.tokenizer = tokenizer
        self.vals = []

        # Approach is to get aspect idx and text idx and push into a list ds

        for i in range(len(self.df)):
            text, aspect, polarity = self.df.iloc[i]
            text_indices, segment_indices = ABSADataset.get_indices(
                text, aspect, self.tokenizer)
            polarity = int(polarity)
            data = {
                "text_indices": text_indices,
                "segment_indices": segment_indices,
                "polarity": polarity,
            }
            self.vals.append(data)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, index):
        return self.vals[index]


if __name__ == "__main__":
    df = pd.read_csv("/srv/home/ahuja/Enterpret/data/train.csv")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = ABSADataset(df, tokenizer)
    loader = DataLoader(
        ds,
        shuffle=True,
        batch_size=32
    )
