import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

x_train = np.load('x_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
x_test = np.load('x_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

print(x_train[:2])

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# model = tf.keras.utils.multi_gpu_model(model, gpus=4)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# model.fit(x_train, y_train, batch_size=128, epochs=10,validation_split=VALIDATION_SPLIT)
log_dir = "logs/fit2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit(x_train, y_train, batch_size=128, epochs=10,validation_split=VALIDATION_SPLIT, callbacks=[tensorboard_callback])
model.fit(x_train, y_train, batch_size=128,epochs=2)

x_test = tokenizer.texts_to_sequences(x_test)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
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
