import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import make_dataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class nn_model(torch.nn.Module):

    def __init__(self):
        super(nn_model, self).__init__()
        width = 64
        act = torch.nn.GELU()
        self.lay1 = torch.nn.Linear(64, width)
        self.activation1 = act
        self.lay2 = torch.nn.Linear(width, width)
        self.activation2 = act
        self.lay3 = torch.nn.Linear(width, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.activation1(x)
        x = self.lay2(x)
        x = self.activation2(x)
        x = self.lay3(x)
        x = self.softmax(x)
        return x


model = nn_model()


class nn_train():

    def __init__(self, file = "data/processed_data.csv", max_len = 64, epochs = 1, learning_rate = 0.01):    
        self.device = torch.accelerator.current_accelerator(
        ).type if torch.accelerator.is_available() else "cpu"
        print(self.device)
        self.model = model.to(self.device)
        self.file = file
        self.max_len = max_len
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate)
        self.tokenizer = TfidfVectorizer(max_features=64)
        
    def read_data(self):
        df = pd.read_csv(self.file, encoding="utf-8")
        df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
        return df
    
    def load_data(self):
        df = self.read_data()
        
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Polarity'])
        
        self.tokenizer.fit(train_data['Processed_Tweets'])
        
        train_set = make_dataset(train_data, self.tokenizer)
        test_set = make_dataset(test_data, self.tokenizer)
        
       

        train_params = {'batch_size': 32,
                        'shuffle': True,
                        'num_workers': 2
                        }

        test_params = {'batch_size': 32,
                       'shuffle': True,
                       'num_workers': 2
                       }

        self.train_dataloader = DataLoader(train_set, **train_params)
        self.test_dataloader = DataLoader(test_set, **test_params)

    def train(self):
        self.model.train()
        print("training mode = ", self.model.training)

        for epoch in range(self.epochs):
            i = 0
            for batch in (self.train_dataloader):
                ids = batch["tweets"].to(self.device)
                targets = batch["targets"].to(self.device)


                self.optimizer.zero_grad()
                output = self.model(ids.float())
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                
                i += 1
                completion = int(round((i/len(self.train_dataloader))*100, 0))
                if completion % 10 < 1:
                    print(f"batch: {i}, completion: {completion} %")
                   
            print(f"Epoch: {epoch+1}, Loss:  {loss.item()}")

        self.model.eval()
    
    def save_model(self):
        print("training mode = ", self.model.training)
        torch.save(self.model.state_dict(), "data/model.pth")
        print("Model saved")
        
    
    def evaluate_model(self):
        self.test_model = nn_model().to(self.device)
        self.test_model.load_state_dict(torch.load("data/model.pth"))
        self.test_model.eval()
        
        
        y_pred = []
        y_test = []
        
        for batch in self.test_dataloader:
            ids = batch['tweets'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            output = self.test_model(ids)
            
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_test.extend(targets.tolist())
            
        con_mat = confusion_matrix(y_test,y_pred)
        print("Confusion Matrix:")
        print(con_mat)
        true_posi = con_mat[1][1]
        false_posi = con_mat[0][1]
        false_neg = con_mat[1][0]
        true_neg = con_mat[0][0]
        accuracy = (true_posi + true_neg) / (true_posi + true_neg + false_posi + false_neg)
        print(f"Accuracy: {round(accuracy, 2)}")
        recall = true_posi / (true_posi + false_neg)
        print(f"Recall: {round(recall, 2)}")
        precision = true_posi / (true_posi + false_posi)
        print(f"Precision: {round(precision, 2)}")
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {round(f1_score, 2)}")
        
        
        
    def IO(self, text, input_model):
        model = nn_model().to(self.device)
        model.load_state_dict(torch.load(input_model))
        model.eval()
        
        
        text = [text]
        text = self.tokenizer.transform(text)
        if hasattr(text, "toarray"):
            text = text.toarray()  
            
        text = torch.tensor(text, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            output_torch = model(text)
            print(output_torch[0][1])
            output = output_torch.cpu().detach().numpy()
        
        if abs(output[0][1]-output[0][0]) < 0.35:
            return "Neutral"
        
        elif output[0][1] > output[0][0]:
            return "Positive"
        
        elif output[0][0] > output[0][1]:
            return "Negative"
        
        
    
if __name__ == "__main__":
    train = nn_train()
    train.load_data()
    train.train()
    train.save_model()
    train.evaluate_model()
        