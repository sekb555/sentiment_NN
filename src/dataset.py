import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class make_dataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.tweets = dataframe["Processed_Tweets"]
        self.targets = dataframe["Polarity"]
        self.max_len = max_len
        
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        tweets = str(self.tweets[index])

        inputs = self.tokenizer.encode_plus(
            tweets,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }