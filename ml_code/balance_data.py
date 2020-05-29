import random
import numpy as np
from tqdm import tqdm
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

data_frame = pd.read_csv(
    './boardgamegeek-reviews/bgg-13m-reviews.csv', index_col=0)
data_frame.drop(data_frame.columns[4], axis=1, inplace=True)
data_frame.drop(data_frame.columns[3], axis=1, inplace=True)
data_frame.drop(data_frame.columns[0], axis=1, inplace=True)


data_frame = data_frame[~data_frame.comment.str.contains("NaN", na=True)]
print(data_frame.head())
print('data shape: ', data_frame.shape)

data_frame["rating"] = data_frame["rating"].round(0).astype(int)
print(data_frame.groupby(["rating"]).count())

rating_subset = data_frame[data_frame['rating'] == 1]
balance_df = rating_subset.sample(20000)

for i in range(9):
    rating_subset = data_frame[data_frame['rating'] == (i+2)]
    r = rating_subset.sample(20000)
    balance_df = balance_df.append(r)

print(balance_df.groupby(["rating"]).count())

nltk.download('punkt')
nltk.download('stopwords')

x = np.array(balance_df.comment)
y = np.array(balance_df.rating)

all_texts = []
for index, text in tqdm(enumerate(x)):
    # lower case
    text = text.lower()
    # tokenize
    words = word_tokenize(text)
    # topwords
    words = [w for w in words if w not in stopwords.words('english')]
    # remove punctuation
    words = [w for w in words if w not in string.punctuation]
    # Stemming
    words = [PorterStemmer().stem(w) for w in words]
    all_texts.append(words)

x = np.array(all_texts)
index = [i for i in range(len(x))]
random.shuffle(index)
x = x[index]
y = y[index]

np.save('balance_x.npy', x)
np.save('balance_y.npy', y)

