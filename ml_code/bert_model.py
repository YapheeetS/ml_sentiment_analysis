from keras_bert import AdamWarmup, calc_train_steps
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
import sklearn.metrics as metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

device_lib.list_local_devices()

SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5
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


token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
indices = []
for text in x_train:
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    indices.append(ids)

items = list(zip(indices, y_train))
np.random.shuffle(items)
indices, sentiments = zip(*items)
indices = np.array(indices)
mod = indices.shape[0] % BATCH_SIZE
if mod > 0:
    indices, sentiments = indices[:-mod], sentiments[:-mod]
x_train = [indices, np.zeros_like(indices)]
y_train = np.array(sentiments)


indices = []
for text in x_test:
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    indices.append(ids)

items = list(zip(indices, y_test))
np.random.shuffle(items)
indices, sentiments = zip(*items)
indices = np.array(indices)
mod = indices.shape[0] % BATCH_SIZE
if mod > 0:
    indices, sentiments = indices[:-mod], sentiments[:-mod]
x_test = [indices, np.zeros_like(indices)]
y_test = np.array(sentiments)

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=1, activation='sigmoid')(dense)
model = keras.models.Model(inputs, outputs)

total_steps, warmup_steps = calc_train_steps(
    num_example=x_train[0].shape[0],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    warmup_proportion=0.1,
)
optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-4, min_lr=LR)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
model.summary()

sess = K.get_session()
uninitialized_variables = set(
    [i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0]
     in uninitialized_variables]
)
sess.run(init_op)

model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    # validation_split=VALIDATION_SPLIT
)

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
