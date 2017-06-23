 # -*- coding: utf-8 -*-
'''

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python

'''

from __future__ import print_function
import numpy as np
np.random.seed(3435)  # for reproducibility, should be first


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, \
    Embedding, Conv1D, MaxPooling1D, AveragePooling1D, \
    Input, Dense, merge
from keras.regularizers import l2
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.constraints import maxnorm
from keras.datasets import imdb
from keras import callbacks
from keras.utils import generic_utils
from keras.models import Model
from keras.optimizers import Adadelta, Adagrad
import time


batch_size = 50
nb_filter = 200
filter_length = 4
hidden_dims = nb_filter * 2
nb_epoch = 60
RNN = GRU
rnn_output_size = 100
folds = 10

print('Loading data...')

from process_sst2_data import SentimentPhrase
from sst2_data import load_data

X_train, y_train, X_test, y_test, W, W2 = load_data()
maxlen = X_train.shape[1]
max_features = len(W)
embedding_dims = len(W[0])

print('Train...')
accs = []
first_run = True
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
rand_idx = np.random.permutation(range(len(X_train)))
X_train = X_train[rand_idx]
y_train = y_train[rand_idx]


def build_model():
    main_input = Input(shape=(maxlen, ), dtype='int32', name='main_input')
    embedding  = Embedding(max_features, embedding_dims,
                  weights=[np.matrix(W)], input_length=maxlen,
                  name='embedding')(main_input)

    embedding = Dropout(0.50)(embedding)

    conv4 = Conv1D(filters=nb_filter,
                          kernel_size=4,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='conv4')(embedding)
    maxConv4 = MaxPooling1D(pool_size=2,
                             name='maxConv4')(conv4)

    conv5 = Conv1D(filters=nb_filter,
                          kernel_size=5,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='conv5')(embedding)
    maxConv5 = MaxPooling1D(pool_size=2,
                            name='maxConv5')(conv5)

    x = merge([maxConv4, maxConv5], mode='concat')

    x = Dropout(0.15)(x)

    x = RNN(rnn_output_size)(x)

    x = Dense(hidden_dims, activation='relu', kernel_initializer='he_normal',
              kernel_constraint = maxnorm(3), bias_constraint=maxnorm(3),
              name='mlp')(x)

    x = Dropout(0.10, name='drop')(x)

    output = Dense(1, kernel_initializer='he_normal',
                   activation='sigmoid', name='output')(x)

    model = Model(input=main_input, output=output)
    model.compile(loss={'output':'binary_crossentropy'},
               	optimizer=Adadelta(lr=0.95, epsilon=1e-06),
               	# optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                # optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4),
                metrics=["accuracy"])
    return model

model = build_model()
if first_run:
    first_run = False
    print(model.summary())

best_val_acc = 0
best_test_acc = 0
for j in xrange(nb_epoch):
    a = time.time()
    his = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    validation_data=[X_test, y_test],
                    shuffle=True,
                    epochs=1, verbose=1)
    print('Epoch %d/%d\t%s' % (j + 1, nb_epoch, str(his.history)))
    if his.history['val_acc'][0] >= best_val_acc:
        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=batch_size,
                                    verbose=2)
        best_val_acc = his.history['val_acc'][0]
        best_test_acc = acc
        print('Got best epoch  best val acc is %f test acc is %f' %
              (best_val_acc, best_test_acc))
        if len(accs) > 0:
            print('Current avg test acc:', str(np.mean(accs)))
    b = time.time()
    cost = b - a
    left = (nb_epoch - j - 1)
    print('One round cost %ds, %d round %ds %dmin left' % (cost, left,
                                                           cost * left,
                                                           cost * left / 60.0))
accs.append(best_test_acc)
print('Avg test acc:', str(np.mean(accs)))
