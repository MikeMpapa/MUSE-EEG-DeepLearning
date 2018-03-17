from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
import keras
import sys, glob
import numpy as np
import random


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def Preprocessing(file,height,width):
    
    data = np.load(file).astype('float')
    
    if data.shape[0] <  height:
        for k in range(height - data.shape[0]):
            data = np.vstack((data,[0]*width))
    elif data.shape[0] >  height:
        data = data[:50,:]

    return [data]





num_classes = 1
im_h = 50
im_w = 60
batch_size = None
epochs = 2000

history = AccuracyHistory()

#LOAD TRAINING DATA
train_success = '../../EEG_DATA_DEEP/SvF/fold_0/train/success/'
train_fail = '../../EEG_DATA_DEEP/SvF/fold_0/train/fail/'

train_success_paths = [filename for filename in glob.glob(train_success+'*npy')] 

train_fail_paths = [filename for filename in glob.glob(train_fail+'*npy')] 


#Load Training Images
train_success_ims = []
for i in train_success_paths:
    x = Preprocessing(i,im_h,im_w)
    train_success_ims.append(x)
train_success_labels = [1]*len(train_success_ims)

train_fail_ims = []
for i in train_fail_paths:
    x = Preprocessing(i,im_h,im_w)
    train_fail_ims.append(x)
train_fail_labels = [0]*len(train_fail_ims)

train_data = train_success_ims + train_fail_ims
train_labels = train_success_labels + train_fail_labels

tmp = list(zip(train_data, train_labels))
random.shuffle(tmp)
train_data, train_labels = zip(*tmp)


x_train = np.array(train_data)
y_train = np.array(train_labels)

########################-----LOAD TESTING DATA------########################################

test_success = '../../EEG_DATA_DEEP/SvF/fold_0/test/success/'
test_fail = '../../EEG_DATA_DEEP/SvF/fold_0/test/fail/'

test_success_paths = [filename for filename in glob.glob(test_success+'*npy')] 
test_fail_paths = [filename for filename in glob.glob(test_fail+'*npy')] 

print len(train_success_paths),len(test_success_paths),len(train_fail_paths),len(test_fail_paths)
#sys.exit()


test_success_ims = []
for i in test_success_paths:
    x = Preprocessing(i,im_h,im_w)
    test_success_ims.append(x)
test_success_labels = [1]*len(test_success_ims)


test_fail_ims = []
for i in test_fail_paths:
    x = Preprocessing(i,im_h,im_w)
    test_fail_ims.append(x)
test_fail_labels = [0]*len(test_fail_ims)



test_data = test_success_ims + test_fail_ims
test_labels = test_success_labels + test_fail_labels

tmp = list(zip(test_data, test_labels))
random.shuffle(tmp)
test_data, train_labels = zip(*tmp)

x_test = np.array(test_success_ims + test_fail_ims)
y_test = np.array(test_success_labels + test_fail_labels)

X_batch, y_batch = ImageDataGenerator().flow(x_train, y_train, batch_size=32)

x_test = x_train
y_test = y_train






print x_train.shape,x_test.shape
print y_train.shape,y_test.shape

#for i in y_train:
 #   print i

#sys.exit()

'''

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(1,im_h,im_w), data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])



model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history],shuffle=True)




score = model.evaluate(x_test, y_test, verbose=0)
'''


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(1,im_h, im_w), data_format="channels_first"))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='sigmoid'))
model.add(Conv2D(64, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-1, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history],shuffle=True)

score = model.evaluate(x_test, y_test)




print('Test loss:', score[0])
print('Test accuracy:', score[1])