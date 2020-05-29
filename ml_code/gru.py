import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
import datetime
from sklearn.model_selection import train_test_split
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print('load data')

x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train, y_train = x, y

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)

fn = 'tokenizer.pkl'
with open(fn, 'wb') as f:
    picklestring = pickle.dump(tokenizer, f)


x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=MAX_SEQUENCE_LENGTH)


num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.GRU(256, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# model.fit(x_train, y_train, batch_size=128, epochs=10,validation_split=VALIDATION_SPLIT)
model.fit(x_train, y_train, batch_size=128, epochs=5)
# model save
model.save('gru_model.h5')
