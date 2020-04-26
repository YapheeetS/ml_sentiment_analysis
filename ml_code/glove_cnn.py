import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

PRETRAINED_DIR = 'pretrained_model'
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

x_train = np.load('x_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
x_test = np.load('x_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)


embeddings_index = {}
f = open(os.path.join(PRETRAINED_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)


# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

model = tf.keras.models.Sequential()
'''
Embedding
input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
output_dim: int >= 0. Dimension of the dense embedding.
embeddings_initializer: Initializer for the embeddings matrix.
'''
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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
