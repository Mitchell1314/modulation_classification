import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
#################################################load datasets#####################################################
Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'),encoding='bytes')
#print(Xd[(b'WBFM', -6)].shape)
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
################obtain snr ranging from -20dB to 18dB##############################
####################obtain moulation methods####################################
X = []
lbl = []
#################################get IQ data for each snr and modulation methods####################################
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])      ###############1000samples for per snr and modulation methods
        for i in range(Xd[(mod,snr)].shape[0]):
                    lbl.append((mod,snr))
X = np.vstack(X)
print("*********************")
print(X.shape)
#print(X[0])
#####################################################split datasets################################################
from sklearn.model_selection import train_test_split
np.random.seed(2016)
n_examples = X.shape[0]
n_train = 160000
train1_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)

#print(train_idx)
X_train1 = X[train1_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train1 = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train1_idx)))
print(Y_train1.shape[0])
X_train,X_test1,Y_train,Y_test1=train_test_split(X_train1,Y_train1,test_size=0.4,random_state=0)
print(X_train.shape[0])
print(Y_test1.shape[0])
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(X_test1.shape)
####################################################################################################################
###############################################build the network####################################################
in_shp = list(X_train.shape[1:])
print(in_shp)
classes = mods
dr = 0.6 # dropout rate (%)
model = models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp))
#model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1"))
model.add(Dropout(0.2))
#model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu',  name="dense1"))
model.add(Dropout(0.3))
#model.add(Dropout(0.2))
model.add(Dense( len(classes),  name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *
adam0=Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam0,metrics=['accuracy'])
nb_epoch = 95   # number of epochs to train on
batch_size = 128 # training batch size
###################################################train the network#################################################

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(X_test1, Y_test1),

   )
# we re-load the best weights once training is finished
model.save('mod_recgnition_1.h5')
#score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
#print(score)
print(history.history['loss'])
plt.figure()
plt.title('Training performance')
plt.plot(history.history['loss'], label='train loss+error')
plt.plot(history.history['val_loss'], label='val_error')

plt.show()
acc = {}
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1




plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.show()
