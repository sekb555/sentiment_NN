import html
import pandas as pd
import numpy as np
import re



class PreprocessData:


    def __init__(self):
        pass


# preprocess the text data

    def preprocess_text(self, text):

        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'@\w+', '', text)  # remove user tags
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
            text = text.strip()
            text = html.unescape(text)
            return text
        elif isinstance(text, pd.Series):
            text = text.str.lower()
            text = text.astype(str).str.replace(
                r'@\w+', '', regex=True)  # remove user tags
            text = text.astype(str).str.replace(
                r'http\S+', '', regex=True)  # remove URLs
            text = text.astype(str).str.replace(
                r'[^a-zA-Z0-9 ]', '', regex=True)  # remove special characters
            # remove leading and trailing whitespaces
            text = text.astype(str).str.strip()
            text = text.apply(html.unescape)  # remove html
            return text


# process the date column for more manageable data
    def process_date(self, date):
        date = date.astype(str).str.split()
        date = pd.DataFrame(date.tolist(), columns=[
                            'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
        date = date.drop(columns=['Timezone', 'Time'])
        return date

# preprocess the training large set of training data
    def preprocess_trainingdata(self, file = "/Users/sekb/Desktop/BERT_sentiment/BERT_sentiment/data/training.1600000.processed.noemoticon.csv"):
        # read the training data and assign the columns names
        df = pd.read_csv(
            file, header=None, encoding="ISO-8859-1")
        df.columns = ["Polarity", "ID", "Date", "Flag", "User", "Tweet"]

        df = df.dropna(subset=["Tweet"])  # remove rows with missing tweets

        df.drop(columns=["ID", "Flag", "User", "Date"], inplace=True)

        df1 = df.head(1024)
        df2 = df.tail(1024)
        df = pd.concat([df1, df2])
        
        # assign text and sentiment to variables
        twts = df["Tweet"]
        sentiments = df["Polarity"]
        sentiments = (sentiments == 4).astype(
            int)  # 0 for negative, 1 for positive
        df["Polarity"] = sentiments

        # preprocess the input data
        df['Processed_Tweets'] = self.preprocess_text(
            twts)
        
        # remove the original tweet column
        df.drop(columns=["Tweet"], inplace=True)

        df.to_csv("data/processed_data.csv", index=False)
        print("Data Preprocessing Complete")


if __name__ == "__main__":
    ppd = PreprocessData()
    ppd.preprocess_trainingdata()