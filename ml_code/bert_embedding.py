from tensorflow.python.client import device_lib
import keras.backend as K
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint
import codecs
import os
from keras_radam import RAdam
import numpy as np
import keras
from keras_bert import Tokenizer
from tqdm import tqdm
from keras_bert import extract_embeddings
import sklearn.metrics as metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

device_lib.list_local_devices()

SEQ_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
MAX_SEQUENCE_LENGTH = 128
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


pretrained_path = 'pretrained_model/uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

x_train = np.load('x_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
x_test = np.load('x_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

for i, d in tqdm(enumerate(x_train)):
    sentence = ' '.join(x_train[i])
    x_train[i] = sentence

for i, d in tqdm(enumerate(x_test)):
    sentence = ' '.join(x_test[i])
    x_test[i] = sentence


x_train = extract_embeddings(pretrained_path, x_train)
x_train = np.array(x_train)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=256, input_shape=(MAX_SEQUENCE_LENGTH,768), return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RAdam(LR),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
            # validation_split=VALIDATION_SPLIT
            )

x_test = extract_embeddings(pretrained_path, x_test)
x_test = np.array(x_test)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

results = model.evaluate(x_test, y_test)
print('Test loss:', results[0])
print('Test accuracy:', results[1])

y_pred = model.predict(x_test)
y_pred = y_pred.reshape(y_pred.shape[0],)

for index, l in enumerate(y_pred):
    if l < 0.5:
        y_pred[index] = 0
    else:
        y_pred[index] = 1

print(metrics.classification_report(y_test, y_pred, digits=5))
