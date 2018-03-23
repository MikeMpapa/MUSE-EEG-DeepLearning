import argparse, sys
import train_deepModel as trNet
import keras
import numpy as np


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train' , '--pathTrainData', type=str,required=True, help="path to training data")
    parser.add_argument('-test' , '--pathTestData', type=str,required=True, help="path to test data")    
    parser.add_argument('-imh' , '--imHeight', type=int, default=100,required=False, help="image height in pixes")  
    parser.add_argument('-imw' , '--imWidth', type=int, default=100,required=False, help="image width in pixes")  
    parser.add_argument('-b' , '--batchSize', default=32, type=int, help="input batch size")    
    parser.add_argument('-e' , '--epochs', default=500, type=int, required=False, help="stride window to based on witch mossing values are created at the end of the last recording")    
    parser.add_argument('-es' , '--epochSteps', default=500, type=int, required=False, help="number of steps per epoch")       
    parser.add_argument('-vs' , '--validationSteps', default=500, type=int, required=False, help="number of steps during model vallidation")       
    parser.add_argument('-p' , '--padding', default="constant", choices=["constant","nearest","reflect","wrap"], type=str,  help="padding out-of-bounds features") 
    parser.add_argument('-c' , '--classes', default=1, type=int, required=False, help="total number of classes")  
    parser.add_argument('-net' , '--classifier', default='VGG', choices=["Inception","VGG"], type=str, required=False, help="total number of classes")       
     

    

    args = parser.parse_args()
   
    return args




def preProcessing(im_h,im_w,batch,padding,classes,path_train,path_test,mode='binary'):

  datagen = trNet.ImageGenerator(padding)

  train_generator = datagen.flow_from_directory(
          path_train,
          target_size=(im_h, im_w),
          color_mode = 'rgb',
          #classes = ['success','fail'],
          batch_size=batch,
          class_mode= mode,
          #save_to_dir = '/home/michalis/Documents/Deep_SequenceLearning/EEG_DATA_DEEP/SvF_images/augmented_data/',
          #save_prefix='aug_im',
          #save_format = 'png',
          shuffle = True
          )


  validation_generator = datagen.flow_from_directory(
          path_test,
          target_size=(im_h, im_w),
          color_mode = 'rgb',
          #classes = ['success','fail'],
          batch_size=batch,
          save_to_dir = '../../EEG_DATA_DEEP/SvF_colored_images/fold_0/augmented_data',
          save_prefix = 'aug_',
          save_format = 'png',
          class_mode=mode)




  return train_generator, validation_generator




def scoreModel(model,data_train,data_test,steps):
      print "----ON TRAIN----"
      score = model.evaluate_generator( data_train, steps=steps, max_queue_size=10, workers=1, use_multiprocessing=True)

      print('Train loss:', score[0])
      print('Train accuracy:', score[1])
      print model.metrics_names
      print "----ON TEST----"
      score =  model.evaluate_generator( data_test, steps=steps, max_queue_size=10, workers=1, use_multiprocessing=True)
      print('Test loss:', score[0])
      print('Test accuracy:', score[1])




def trainModel(model,train,test,epochs,e_step,v_step,batch_size):
    #history = AccuracyHistory()
    model.fit_generator(train,
        steps_per_epoch=e_step,
        epochs=epochs,
        validation_data=test,
        validation_steps=v_step,
        verbose=1)
    #print history.acc
    return model





   



def main(net,im_h,im_w,batch_size,epochs,e_step,v_step, padding,classes,path_train,path_test):

  train,test = preProcessing(im_h,im_w,batch_size,padding,classes,path_train,path_test,"binary")
  #for i in test:
   # print i[0].shape

  try:
      if net == 'vgg':
        model = trNet.VGG(im_h ,im_w,classes, batch_size,epochs)
      elif net =="inception": 
        model = trNet.Inception(im_h ,im_w,classes, batch_size,epochs)
  except Exception as e:
    raise e

  trained_model = trainModel(model,train,test,epochs,e_step,v_step,batch_size)
  scoreModel(trained_model,train,test,v_step)
  #print type(train), type(test)
  #print train.shape, test.shape




if __name__ == '__main__':
  args = parseArguments()
  main(args.classifier.lower(),args.imHeight,args.imWidth,args.batchSize,args.epochs,args.epochSteps,args.validationSteps,args.padding,args.classes,args.pathTrainData,args.pathTestData)
  