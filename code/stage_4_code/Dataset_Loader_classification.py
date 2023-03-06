'''
Concrete IO class for a specific dataset
'''
import torch

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from glob import glob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from random import seed
from random import shuffle
import string
from torch import nn


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None


    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def load_subfolder(self, train_or_test, pos_or_neg, toy_size=0):
        folder_path = self.dataset_source_folder_path + train_or_test + '/' + pos_or_neg + '/*.txt'
        file_list = glob(folder_path)
        # In toy testing, trim file_list to speed up testing.
        if toy_size > 0:
            shuffle(file_list)
            file_list = file_list[:toy_size]
        reviews = []
        for file in file_list:
            f = open(file, 'rt')
            text = f.read()
            f.close()

            # Split into words.
            tokens = word_tokenize(text)
            # Convert to lower case.
            tokens = [w.lower() for w in tokens]
            # Remove punctuation from each word.
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # Remove remaining tokens that are not alphabetic.
            words = [word for word in stripped if word.isalpha()]
            # Filter out stop words.
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
            #print(words[:100])
            reviews.append(words)

        return reviews


    def normalize_review_length(self, reviews, normal_length):
        for i in range(len(reviews)):
            if len(reviews[i]) < normal_length:
                reviews[i] = reviews[i] + (['[PAD]'] * (normal_length - len(reviews[i])))
            elif len(reviews[i]) > normal_length:
                reviews[i] = reviews[i][:normal_length]

        return reviews

    '''
    def Word2Vec(self, reviews):
        vocab = set([word for review in reviews for word in review])
        vocab_size = len(vocab)
        embed_dim = 50

        embedding = nn.Embedding(vocab_size, embed_dim)
        
        ...
        
        return
    '''

    def load(self):
        toy_size = 10

        seed(42)

        # Load data
        print('loading data...')
        # Load subfolders
        pos_train_reviews = self.load_subfolder('train', 'pos', toy_size)
        neg_train_reviews = self.load_subfolder('train', 'neg', toy_size)
        pos_test_reviews = self.load_subfolder('test', 'pos', toy_size)
        neg_test_reviews = self.load_subfolder('test', 'neg', toy_size)

        # Normalize review length
        # Get average review length.
        total_words = 0
        for review in pos_train_reviews:
            total_words += len(review)
        for review in neg_train_reviews:
            total_words += len(review)
        for review in pos_test_reviews:
            total_words += len(review)
        for review in neg_test_reviews:
            total_words += len(review)

        avg_review_length = round(total_words / (len(pos_train_reviews) + len(neg_train_reviews) + len(pos_test_reviews) + len(neg_test_reviews)))

        self.normalize_review_length(pos_train_reviews, avg_review_length)
        self.normalize_review_length(neg_train_reviews, avg_review_length)
        self.normalize_review_length(pos_test_reviews, avg_review_length)
        self.normalize_review_length(neg_test_reviews, avg_review_length)

        # Add labels
        pos_train_labels = [1] * len(pos_train_reviews)
        neg_train_labels = [0] * len(neg_train_reviews)
        pos_test_labels = [1] * len(pos_test_reviews)
        neg_test_labels = [0] * len(neg_test_reviews)

        labeled_pos_train = list(zip(pos_train_labels, pos_train_reviews))
        labeled_neg_train = list(zip(neg_train_labels, neg_train_reviews))
        labeled_pos_test = list(zip(pos_test_labels, pos_test_reviews))
        labeled_neg_test = list(zip(neg_test_labels, neg_test_reviews))

        labeled_train = labeled_pos_train + labeled_neg_train
        labeled_test = labeled_pos_test + labeled_neg_test
        shuffle(labeled_train)
        shuffle(labeled_test)

        y_train_tuple, X_train_tuple = zip(*labeled_train)
        y_test_tuple, X_test_tuple = zip(*labeled_test)

        # Word2Vec
        # Collect all unique words.
        vocab = list(set([word for review in X_train_tuple for word in review]))
        # Convert words to numbers by enumerating them and storing the word, num pairs in a dict.
        vocab_dict = {word: index for index, word in enumerate(vocab)}
        embedding_dim = 100
        embedding = nn.Embedding(len(vocab), embedding_dim)

        # Convert the input sentences to tensors
        inputs = [[vocab_dict[word] for word in review] for review in X_train_tuple]
        inputs = torch.LongTensor(inputs)

        X_train = final_embedding(X_train_tuple)
        X_test = final_embedding(X_test_tuple)
        y_train = list(y_train_tuple)
        y_test = list(y_test_tuple)

        '''
        # Convert words to numbers by enumerating them and storing the word, num pairs in a dict.
        vocab_dict = {word: index for index, word in enumerate(vocab)}
        # Store converted words as a tensor.
        vocab_tensor = torch.LongTensor([vocab_dict[word] for word in vocab])
        embed_dim = 50

        #y_train = torch.LongTensor(y_train_tuple)
        #X_train = torch.LongTensor(X_train_tuple)
        #y_test = torch.LongTensor(y_test_tuple)
        #X_test = torch.LongTensor(X_test_tuple)


        embedding = nn.Embedding(vocab_size, embed_dim)
        embedded_vocab = embedding(vocab_tensor)
        print(embedded_vocab.shape)
        '''

        '''
        OK, I have the embedded vocab, but I really need the X_train to be
        encoded.  How do I do that?  The same way.
        I need each word to be embedded before I can use it.
        '''

        #embedding(X_train)
        #embedding(X_test)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
