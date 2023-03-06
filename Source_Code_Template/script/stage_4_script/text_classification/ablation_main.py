import pandas as pd
import nltk
from nltk.corpus import stopwords
from preprocessing import pandas_loader, remove_punc, stemmed_review
import torch
import torchtext # NEEDS VERSION 0.9.1 TO WORK WITH LEGACY!!!
from rnn import RNN
from train import model_training
import matplotlib.pyplot as plt

# Begin by loading in the data
print("Loading data in dataframes, standby...")
train_pos_df = pandas_loader("train", "pos")
print("Positive training data loaded")
train_neg_df = pandas_loader("train", "neg")
print("Negative training data loaded")
test_pos_df = pandas_loader("test", "pos")
print("Positive testing data loaded")
test_neg_df = pandas_loader("test", "neg")
print("Negative training data loaded")

# Bring the dataframes together
training_df = pd.concat([train_pos_df, train_neg_df], axis=0)
testing_df = pd.concat([test_pos_df, test_neg_df], axis=0)

# Get rid of any extra formatting issues
print("Removing unnecessary tabs and new lines...")
training_df = training_df.replace(r'\r+|\n+|\t+',' ', regex=True)
testing_df = testing_df.replace(r'\r+|\n+|\t+',' ', regex=True)

# Remove punctuation, make everything lowercase, and get rid of stopwords
stopwords = stopwords.words("english")
print("Removing punctuation, making everything lowercase, and getting rid of stopwords for training set...")
training_df["Review_Text"] = training_df["Review_Text"].apply(remove_punc)
training_df["Review_Text"] = training_df["Review_Text"].apply(str.lower)
training_df["Review_Text"] = training_df["Review_Text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
print("Removing punctuation, making everything lowercase, and getting rid of stopwords for testing set...")
testing_df["Review_Text"] = testing_df["Review_Text"].apply(remove_punc)
testing_df["Review_Text"] = testing_df["Review_Text"].apply(str.lower)
testing_df["Review_Text"] = testing_df["Review_Text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# Created stemmed text column 
print("Creating stemmed column for training set...")
training_df["Stemmed_Review_Text"] = training_df["Review_Text"].apply(stemmed_review)
print("Creating stemmed column for testing set...")
testing_df["Stemmed_Review_Text"] = testing_df["Review_Text"].apply(stemmed_review)

# Export for later use
print("Preprocessing complete, exporting data...")
testing_df.to_csv("testing_corpus.csv", index=False)
training_df.to_csv("training_corpus.csv", index=False)
print("Export complete")

# Won't use stemming this time around, so let's drop it
print("Creating pure dataset...")
trainingDF = training_df.drop(columns=["Stemmed_Review_Text"])
testingDF = testing_df.drop(columns=["Stemmed_Review_Text"])

# Export this new data as "pure" for future use
trainingDF.to_csv("pure_training.csv", index=False)
testingDF.to_csv("pure_testing.csv", index=False)
print("Exported pure dataset")

# Create fields for review text and label
TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)
Fields = [("REVIEW_TEXT", TEXT), ("isPos", LABEL)]
print("Loading pure dataset into torchtext...")
train_dataset = torchtext.legacy.data.TabularDataset(path="pure_training.csv", format="csv", skip_header=True, fields=Fields)
test_dataset = torchtext.legacy.data.TabularDataset(path="pure_testing.csv", format="csv", skip_header=True, fields=Fields)
print("Data loaded successfully")

# Build vocab for both the TEXT data (max 75k) and LABEL data (Only 2 since "0" or "1")
TEXT.build_vocab(train_dataset, max_size=75000)
LABEL.build_vocab(train_dataset)
print("Creating torchtext data loaders...")
train_loader, test_loader = torchtext.legacy.data.BucketIterator.splits(
    (train_dataset, test_dataset),
    # Gives us 200 total batches
    batch_size=125,
    sort_within_batch=False,
    sort_key=lambda x: len(x.REVIEW_TEXT),
    # Use cuda device
    device=torch.device(0)
)
print("Created torchtext data loaders")

print("Building RNN model...")
# Begin building model
model = RNN(input_dim=len(TEXT.vocab),
            embedding_dim=125,
            hidden_dim=125,
            output_dim=2)

model = model.to(torch.device(0))
optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
print("Completed building RNN model")

# Begin training of model
print("Starting model training...")
model_training(model, optimizer, 25, train_loader, test_loader, torch.device(0))
