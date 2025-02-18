import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from dataset import make_dataset


class nn_model(torch.nn.Module):

    def __init__(self):
        super(nn_model, self).__init__()
        self.lay1 = torch.nn.Linear(64, 45)
        self.activation = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(45, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.activation(x)
        x = self.lay2(x)
        x = self.softmax(x)
        return x


model = nn_model()

print(model)


class nn_train():

    def __init__(self, file, max_len = 64, epochs = 3, learning_rate = 0.001):    
        self.device = torch.accelerator.current_accelerator(
        ).type if torch.accelerator.is_available() else "cpu"
        print(self.device)
        self.model = model.to(self.device)
        self.file = file
        self.max_len = max_len
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=learning_rate)
        
    def read_data(self):
        df = pd.read_csv(self.file, encoding="utf-8")
        df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
        return df
    
    def load_data(self):
        df = self.read_data()

        train_data = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        test_data = df.drop(train_data.index).reset_index(drop=True)

        train_set = make_dataset(train_data, self.tokenizer, self.max_len)
        test_set = make_dataset(test_data, self.tokenizer, self.max_len)

        train_params = {'batch_size': 16,
                        'shuffle': True,
                        'num_workers': 2
                        }

        test_params = {'batch_size': 16,
                       'shuffle': True,
                       'num_workers': 2
                       }

        self.train_dataloader = DataLoader(train_set, **train_params)
        self.test_dataloader = DataLoader(test_set, **test_params)

    def train(self):
        self.model.train()
        
        for epochs in range(self.epochs):
            for batch in self.train_dataloader:
                ids = batch['ids']
                mask = batch['mask']
                token_type_ids = batch['token_type_ids']
                targets = batch['targets']

                self.model.zero_grad()
                output = self.model(ids)
                loss = torch.model.functional.cross_entropy(output, targets)
                loss.backward()
                self.optimizer.step()
                
                print(f"Epoch: {epochs}, Loss:  {loss.item()}")