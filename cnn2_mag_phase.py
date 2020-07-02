import numpy as np


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import operator
import pickle
from numpy import linalg as la

maxlen = 128
snrs = ""
mods = ""
test_idx = ""
lbl = ""


def gendata(fp, nsamples):
    global snrs, mods, test_idx, lbl
    Xd = pickle.load(open(fp, 'rb'),encoding='bytes')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X)

    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.7)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    return (X_train, X_test, Y_train, Y_test)


def norm_pad_zeros(X_train, nsamples):
    print(X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = X_train[i, :, 0] / la.norm(X_train[i, :, 0], 2)
    return X_train


def to_amp_phase(X_train, X_test, nsamples):
    X_train_cmplx = X_train[:, 0, :] + 1j * X_train[:, 1, :]
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, 1, :], X_train[:, 0, :]) / np.pi

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    return (X_train, X_test)


xtrain1, xtest1, ytrain1, ytest1 = gendata("RML2016.10a_dict.pkl", 128)

xtrain1, xtest1 = to_amp_phase(xtrain1, xtest1, 128)

xtrain1 = xtrain1[:, :maxlen, :]
xtest1 = xtest1[:, :maxlen, :]

xtrain1 = norm_pad_zeros(xtrain1, maxlen)
xtest1 = norm_pad_zeros(xtest1, maxlen)

X_train = xtrain1
X_test = xtest1

Y_train = ytrain1
Y_test = ytest1

print("--" * 50)
print("Training data:", X_train.shape)
print("Training labels:", Y_train.shape)
print("Testing data", X_test.shape)
print("Testing labels", Y_test.shape)
print("--" * 50)


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"




def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm, 2)
    print(cm)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = mods
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt
from keras import layers
from keras import Sequential,layers
from keras.layers import Reshape,Conv2D,Dense,Activation,Dropout,Flatten,ZeroPadding2D
from keras import optimizers
from keras import models
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3, mode='auto',factor=0.01)
adam0=optimizers.Adam(lr=0.001)

in_shp=list(X_train.shape[1:])
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(Conv2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1"))
model.add(Dropout(0.3))
model.summary()
model.add(ZeroPadding2D(padding=(2,2)))
model.add(Conv2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2"))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(256, activation='relu', name="dense1"))
model.add(Dropout(0.6))
model.add(Dense( len(mods), name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(mods)]))

model.compile(loss='categorical_crossentropy', optimizer=adam0,metrics=['accuracy'])
nb_epoch = 95    # number of epochs to train on
batch_size = 128  # training batch size
###################################################train the network#################################################

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(X_test, Y_test)
   )
# we re-load the best weights once training is finished
model.save('cnn_mag_pha.h5')
#score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
#print(score)
print(history.history['loss'])
plt.figure()
plt.title('Training performance')
plt.plot(history.history['loss'], label='train loss+error')
plt.plot(history.history['val_loss'], label='val_error')
plt.show()
acc = {}
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    #print("###################################")
    test_SNRs=list(test_SNRs)
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
    #print(snr)
    #print(np.where(np.array(test_SNRs) == snr))
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    #print(test_Y_i_hat)
    conf = np.zeros([len(mods), len(mods)])
    confnorm = np.zeros([len(mods), len(mods)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(mods)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    #plt.figure()
    plot_confusion_matrix(confnorm, labels=mods, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))
    plt.show()

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor


    acc[snr] = 1.0 * cor / (cor + ncor)

plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.subplots_adjust(wspace =0.2, hspace =0.2)
plt.show()



