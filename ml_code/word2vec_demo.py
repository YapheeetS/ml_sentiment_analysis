import numpy as np
import gensim
import logging
from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize, pos_tag

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

sentences = []
for i in range(0, len(documents)):
    sentences.append(word_tokenize(str(documents[i])))

print(sentences)
print(len(sentences))
print(len(documents))

print(len(np.unique(np.concatenate(sentences))))

model = Word2Vec(sentences, size=10, min_count=1, window=5)

# model.save('Word2Vec2.dict')
# new_model = Word2Vec.load('Word2Vec2.dict')
print(model)

# array = [np.ndarray.flatten(new_model[sentences[i]]) for i in range(0, len(documents))]
print(model['Human'])
