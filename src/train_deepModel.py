from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras 




# Data Augmentation and Preprocessing
def ImageGenerator(padding):
    '''
    generator = ImageDataGenerator(featurewise_center=True,
      samplewise_center=True,
      featurewise_std_normalization=True,
      samplewise_std_normalization=True,
      zca_whitening=True,
      zca_epsilon=1e-6,
      rotation_range=10.,
      width_shift_range=0.5,
      height_shift_range=0.5,
      shear_range=0.2,
      zoom_range=0.3, #[1-0.3 , 1+0.3]
      channel_shift_range=0.,
      fill_mode=padding,
      cval=0.,
      horizontal_flip=True,
      vertical_flip=True,
      rescale=None,
      preprocessing_function=None,
      #save_to_dir = '../../EEG_DATA_DEEP/SvF_images/fold_0/augmented_data',
      #save_prefix = 'aug_',
      #save_format = 'png',
      data_format='channels_first')
    '''
    generator = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   zca_whitening=True,
                                   rotation_range=1,
                                   data_format='channels_first')
    return generator






def VGG( im_h, im_w,num_classes, batch_size,epochs, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):

    model = Sequential()

    model.add(Conv2D(32, (7, 7), activation='sigmoid', input_shape=(1,im_h, im_w), data_format="channels_first"))
    model.add(Conv2D(32, (5, 5), activation='sigmoid'))
    model.add(Conv2D(32, (5, 5), activation='sigmoid'))
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

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

    return model