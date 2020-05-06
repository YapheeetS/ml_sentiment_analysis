import re
import os
import numpy as np
from tqdm import tqdm
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')


class Data_preprocess(object):
    def __init__(self):
        pass

    # regular expression
    def rm_tags(self, text):
        re_tag = re.compile(r'<[^>]+>')
        return re_tag.sub('', text)


    def read_files(self, filetype):
        path = "./aclImdb/"
        file_list = []
        pos_num = 0
        neg_num = 0
        positive_path = path + filetype+"/pos/"
        for f in os.listdir(positive_path):
            file_list += [positive_path+f]
            pos_num += 1
        negative_path = path + filetype+"/neg/"
        for f in os.listdir(negative_path):
            file_list += [negative_path+f]
            neg_num += 1
        print('read', filetype, 'files:', len(file_list))
        print('pos_num: ', pos_num)
        print('neg_num: ', neg_num)
        all_labels = ([1] * pos_num + [0] * neg_num)
        all_texts = []
        for index, fi in tqdm(enumerate(file_list)):
            with open(fi, encoding='utf8') as file_input:
                filelines = file_input.readlines()
                if len(filelines) != 0:
                    text = filelines[0]
                    # remove < > tag
                    text = self.rm_tags(text)
                    # lower case
                    text = text.lower()
                    # tokenize
                    words = word_tokenize(text)
                    # topwords
                    words = [w for w in words if w not in stopwords.words('english')]
                    # # Stemming
                    words = [PorterStemmer().stem(w) for w in words]
                    all_texts.append(words)
                else:
                    print('empty index: ', index)
                    all_texts.append([''])
            # if index == 10:
            #     break

        return all_texts, all_labels


data_preprocess = Data_preprocess()
x_train, y_train = data_preprocess.read_files('train')
x_test, y_test = data_preprocess.read_files('test')

train_index = [i for i in range(len(x_train))]
test_index = [i for i in range(len(x_test))]

random.shuffle(train_index)
random.shuffle(test_index)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train[train_index]
y_train = y_train[train_index]
x_test = x_test[test_index]
y_test = y_test[test_index]

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
