import numpy as np
import gensim
import logging
from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize, pos_tag
import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


x_train = np.load('x_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
x_test = np.load('x_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)


# model = Word2Vec(x_train, size=100, min_count=1, window=5)
# model.save('Word2Vec2.dict')
word2vec_model = Word2Vec.load('Word2Vec2.dict')
print(word2vec_model)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=MAX_SEQUENCE_LENGTH)

print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = word2vec_model[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                    mask_zero=True,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(4, strides=4))
model.add(tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(4, strides=4))
model.add(tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(4, strides=4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# print(model.layers[0])
# model.layers[0].trainable = False
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2)
model.fit(x_train, y_train, batch_size=128, epochs=4)

x_test = tokenizer.texts_to_sequences(x_test)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen=MAX_SEQUENCE_LENGTH)
score = model.evaluate(x_test, y_test, verbose=0)
print('accuracy score: ', score)

y_pred = model.predict(x_test)
y_pred = y_pred.reshape(y_pred.shape[0],)

for index, l in enumerate(y_pred):
    if l < 0.5:
        y_pred[index] = 0
    else:
        y_pred[index] = 1

print(metrics.classification_report(y_test, y_pred, digits=5))
