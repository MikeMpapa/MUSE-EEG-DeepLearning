from keras.layers import *
from keras.optimizers import *
from keras.models import *
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
                                   #rotation_range=1,
                                   #height_shift_range=0.2,
                                   rescale=0.3,
                                   fill_mode=padding,
                                   cval =0,
                                   vertical_flip=True,
                                   data_format='channels_last')
    return generator






def VGG( im_h, im_w,num_classes, batch_size,epochs, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(im_h, im_w,1), data_format="channels_last"))
    model.add(Conv2D(32, (3, 3), activation='sigmoid'))
    #model.add(Conv2D(32, (3, 3), activation='tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='sigmoid'))
    #model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=0.000001,  momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

    return model


def Inception( im_h, im_w,num_classes, batch_size,epochs, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',input_shape=(im_h, im_w,3))
    #top_model = Sequential()
     # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='tanh')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers:
        layer.trainable = False
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])


    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(1, activation='sigmoid'))
    #model.add(top_model)
    #model.compile(loss='binary_crossentropy',
     #         optimizer=SGD(lr=1e-4, momentum=0.9),
    #metrics=['accuracy'])
    return  model