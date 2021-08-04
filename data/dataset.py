import string

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    def preprocess(self, df):
        df['text'] = df['text'].apply(lambda x: str(x).lower())
        df['label'] = df['label'].apply(lambda x: int(x))
        df['aspect'] = df['aspect'].apply(lambda x: str(x).lower())
        df['text'] = df['text'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.punctuation)))  # removing punctuation
        df['aspect'] = df['aspect'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.punctuation)))  # removing punctuation
        df['text'] = df['text'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.digits)))  # removing integers
        df['aspect'] = df['aspect'].apply(lambda x: str(x).translate(
            str.maketrans('', '', string.digits)))  # removing integers
        return df

    def __init__(self, df, tokenizer):
        self.df = self.preprocess(df)
        self.tokenizer = tokenizer
        self.vals = []

        # Approach is to get aspect idx and text idx and push into a list ds

        for i in range(len(self.df)):
            text, aspect, polarity = self.df.iloc[i]
            text_len = len(text.split())
            aspect_len = len(aspect.split())
            text_indices = self.tokenizer(
                str(text),
				str(aspect),
                padding='max_length',
                max_length=350,  # Max length assuming
                return_tensors='pt',
                add_special_tokens=True,
            )

            segment_indices = ([0] * (text_len + 2) + [1]*(aspect_len + 1))
            segment_indices = torch.LongTensor(
                pad_and_truncate(segment_indices, 350))
            # aspect_indices = self.tokenizer(
            # 	str(aspect),
            # 	padding=True,
            # 	truncation=True,
            # 	max_length=400,  # Max length assuming
            # 	return_tensors='pt'
            # )
            polarity = int(polarity)
            data = {
                "text_indices": text_indices['input_ids'].squeeze(0),
                "segment_indices": segment_indices,
                "polarity": polarity,
            }
            self.vals.append(data)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, index):
        return self.vals[index]


if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = ABSADataset(df, tokenizer)
    loader = DataLoader(
		ds,
		shuffle = True,
		batch_size = 32
	)
    # tokenizer.convert_ids_to_tokens()
    # print(tokenizer.decode(ds[0]['text_indices']))
    for i, data in enumerate(loader):
        if i == 5:
            break
        print(data)
    # print(loader[:10])
    # for i in range(len(ds)):
    #     tokens = (ds[i]['text_indices'])
    #     tmp = ' '.join(tokenizer.convert_ids_to_tokens(tokens))
    #     print(tmp)
