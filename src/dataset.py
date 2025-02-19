import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class make_dataset(Dataset):
    
    def __init__(self, dataframe, tokenizer):
        self.tweets = dataframe["Processed_Tweets"]
        self.targets = dataframe["Polarity"].values
        self.tokenizer = tokenizer
        
        self.tweets = self.tokenizer.transform(self.tweets)
        self.tweets = self.tweets.toarray() if hasattr(self.tweets, "toarray") else self.tweets
        
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):

        return {
            'tweets': torch.tensor(self.tweets[index], dtype=torch.float),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }