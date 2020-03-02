import re
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# regular expression
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
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
    for fi in tqdm(file_list):
        with open(fi, encoding='utf8') as file_input:
            filelines = file_input.readlines()
            text = filelines[0]
            # # remove < > tag
            # text = rm_tags(text)
            # # lower case
            # text = text.lower()
            # # tokenize
            # words = word_tokenize(text)
            # # topwords
            # words = [w for w in words if w not in stopwords.words('english')]
            # # Stemming
            # words = [PorterStemmer().stem(w) for w in words]
            all_texts.append(text)

    return all_texts, all_labels

x_train, y_train = read_files('train')
# x_test, y_test = read_files('test')

print(y_train[3])
print(x_train[3])
