import pandas as pd
import glob
from nltk.stem.porter import PorterStemmer

def pandas_loader(type, folder):
    if folder == "pos":
        num = 1
    elif folder == "neg":
        num = 0
    folder_path = type + "/" + folder + "/*"
    file_list = glob.glob(folder_path)
    with open(file_list[0]) as f:
        first_string = f.readlines()
    f.close()
    data = {'Review_Text': first_string, 'isPos': num}
    df = pd.DataFrame(data)

    for i in range(1, len(file_list)):
        with open(file_list[i]) as f:
            temp_string = f.readlines()
        f.close()
        temp_data = {'Review_Text': temp_string, 'isPos': num}
        temp_df = pd.DataFrame(temp_data)
        df = pd.concat([df, temp_df], axis=0)
    
    return df

def remove_punc(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', ",", "(", ")", "-", "<", "/><br", "br", "/>br", "'", "'s"))
    return final

def stemmed_review(review):
    stemmer = PorterStemmer()
    tokens = review.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)