'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
from collections import Counter
import torchtext
import math


import nltk
from nltk.corpus import stopwords

import re



class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')


        # preprocessed_df = df.copy()
        # #convert all uppercase words to lower-case
        # preprocessed_df['Joke'] = preprocessed_df['Joke'].str.lower()
        # #remove all words inside "()" and "[]" and also remove double-spaces
        # preprocessed_df['Joke'] = preprocessed_df['Joke'].str.replace(r'\([^)]*\)|\[[^]]*\]', '')
        # preprocessed_df['Joke'] = preprocessed_df['Joke'].str.replace(r'\s{2,}', ' ')
        # preprocessed_df['Joke'] = preprocessed_df['Joke'].str.replace(r'[,.!?*";)(:](?!\d)', '')
        #
        # unique_words = set()
        # for index, row in preprocessed_df.iterrows():
        #     # access the text string in the row
        #     text = row['Joke']
        #
        #     # split the text string into individual words
        #     words = text.split()
        #
        #     # add the words to the set of unique words
        #     unique_words.update(words)
        #
        # unique_words.add(".")
        # # print the set of unique words
        # index_to_word = {index: word for index, word in enumerate(unique_words)}
        # word_to_index = {word: index for index, word in enumerate(unique_words)}
        #
        #
        # # #transform a df that contains words into a tokenized dataframe
        # # df['tokens'] = preprocessed_df['Joke'].str.split().map(lambda x: [word_to_index[word] for word in x])
        # # # transform a df that was tokenized back into words
        # # df['text_from_tokens'] = df['tokens'].map(lambda x: ' '.join([index_to_word[index] for index in x]))
        #
        # # print(df['Joke'].iloc[0])
        # # print(df['tokens'].iloc[0])
        # # print(df['text_from_tokens'].iloc[0])
        # preprocessed_df['Joke'] = preprocessed_df['Joke'].str.split().map(lambda x: [word_to_index[word] for word in x])
        # X = preprocessed_df.drop(columns='ID')
        #
        # print(X.head())

        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)

        #make a copy of the dataframe to be manipulated
        preprocessed_df = df.copy()

        #Make undercase, remove all words inside "()" and "[]" and also remove double-spaces
        preprocessed_df['Joke'] = preprocessed_df['Joke'].str.lower()
        preprocessed_df['Joke'] = preprocessed_df['Joke'].str.replace(r'\([^)]*\)|\[[^]]*\]', '', regex=True)

        #remove occurences of doublespaces
        preprocessed_df['Joke'] = preprocessed_df['Joke'].str.replace(r'\s{2,}', ' ', regex=True)
        preprocessed_df['Joke'] = preprocessed_df['Joke'].str.replace(r'[^\w\s]+', '', regex=True)

        #splits the input text into tokens, in this case single words
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        tokenized_data = [tokenizer(sentence) for sentence in preprocessed_df['Joke']]

        #create a vocab list that contains all unique tokens.
        #we also define a padding, unknown, start of sentence, and end of sentence tokens we can use.
        #all tokens that appear less than 10 times will be replaced by <unk>
        vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_data, min_freq=10,
                                                           specials=['<pad>', '<unk>', '<sos>', '<eos>'])
        # unique_tokens = set([token for sentence in tokenized_data for token in sentence])
        # print(len(unique_tokens))
        # print(len(vocab))
        #
        # #average sentence length used as max_length of our data
        # avg_length = sum(len(sentence) for sentence in tokenized_data) / len(tokenized_data)
        # print(f"Average sentence length: {avg_length}")
        # avg_length = math.ceil(avg_length)
        # max_length = avg_length  # define the maximum length for a sentence
        #
        # #transform data from tokens into their numerical equivalent.
        # #For example if <unk> is labeled as 0 is the most common token
        # numerical_data = []
        # for tokenized_sentence in tokenized_data:
        #     numerical_sentence = []
        #     for token in tokenized_sentence:
        #         if token in vocab:
        #             numerical_sentence.append(vocab[token])
        #         else:
        #             numerical_sentence.append(vocab['<unk>'])
        #     numerical_sentence.append(vocab['<eos>'])  # append <eos> token
        #
        #     # pad the sentence if its length is less than the maximum length
        #     if len(numerical_sentence) < max_length:
        #         pad_length = max_length - len(numerical_sentence)
        #         numerical_sentence.extend([vocab['<pad>']] * pad_length)
        #     # truncate the sentence if its length is greater than the maximum length
        #     else:
        #         numerical_sentence = numerical_sentence[:max_length]
        #         numerical_sentence[-1] = vocab['<eos>']  # update the <eos> token
        #
        #     numerical_data.append(numerical_sentence)
        #
        # total_tokens = sum(len(sentence) for sentence in tokenized_data)
        # print(f"Total number of tokens: {total_tokens}")
        #this code lets us translate numerical data into a readable form
        # itos = vocab.get_itos()
        # for i in range(3):
        #     string_data = [itos[token] for token in numerical_data[i]]
        #     print(" ".join(string_data))

        # text = preprocessed_df['Joke'].str.cat(sep=' ')
        # words = text.split(' ')

        # #order all words encountered in the csv by frequency.
        # word_counts = Counter(words)
        # unique_words = sorted(word_counts, key=word_counts.get, reverse=True)
        # print(len(unique_words))
        # print(unique_words)
        #
        # index_to_word = {index: word for index, word in enumerate(unique_words)}
        # word_to_index = {word: index for index, word in enumerate(unique_words)}
        #
        # words_indexes = [word_to_index[w] for w in words]


        return {'tokenized_data':  tokenized_data, 'vocabulary': vocab}


