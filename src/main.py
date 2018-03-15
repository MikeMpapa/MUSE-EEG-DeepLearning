import os, argparse,sys,pdb
import numpy as np
import time
from tqdm import  tqdm


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e' , '--experimentType', type=int, choices = [0,1],required=True, help="0: Build SuccessVSFail Dataset - 1: Build FinalScorePrediction Dataset")  
    parser.add_argument('-f' , '--foldPath', required=True, help="path to the root of the folds")  
    parser.add_argument('-wh' , '--imageHeight', default=50 ,type=int,  help="vertical dimention of input frame  -image height-")  
    parser.add_argument('-ws' , '--windowStride', default=5, type=int, help="stride window to split the sequencial frames of a recording")    
    parser.add_argument('-s' , '--smoothingStride', default=3, type=int, help="stride window to basedd on witch mossing values are created at the end of the last recording")    
    parser.add_argument('-p' , '--padding', default=0, type=int, choices = [0,1], help="0: zerro padding - 1: mean padding")    

    args = parser.parse_args()
   
    return args

    
   

#Recreate features to repeat the Success VS Fail paper with DL
def CreateDataset_SvF(original_datapath):
    for fold in range(10):
        samples_in_rec = []
        fnames = []
        
        train_datapath = original_datapath+'/fold_'+str(fold)+'/train' 
        test_datapath = original_datapath +'/fold_'+str(fold)+'/test' 
        
        train_success_samples = os.listdir(train_datapath+"/success") 
        train_fail_samples = os.listdir(train_datapath+"/fail") 

        test_success_samples = os.listdir(test_datapath+"/success") 
        test_fail_samples = os.listdir(test_datapath+"/fail") 


        for user in tqdm(range(users)):
            for session in range(mx_num_of_sessions):
                for trial in range(rounds):
                    filename = ('_').join((str(user+1),str(session+1),str(trial+1))) 
                    fnames.append(filename)
                    
                    if filename+'.npz' in train_success_samples:
                        output = '../../EEG_DATA_DEEP/SvF/fold_'+str(fold)+'/train/success'
                        tmp_path = ('/').join((train_datapath,"success",filename+'.npz')) 
                    elif filename+'.npz' in train_fail_samples:
                        output = '../../EEG_DATA_DEEP/SvF/fold_'+str(fold)+'/train/fail'
                        tmp_path = ('/').join((train_datapath,"fail",filename+'.npz')) 
                    elif filename+'.npz' in test_success_samples:
                        output = '../../EEG_DATA_DEEP/SvF/fold_'+str(fold)+'/test/success'
                        tmp_path = ('/').join((test_datapath,"success",filename+'.npz')) 
                    elif filename+'.npz' in test_fail_samples:
                        output = '../../EEG_DATA_DEEP/SvF/fold_'+str(fold)+'/test/fail'
                        tmp_path = ('/').join((test_datapath,"fail",filename+'.npz')) 
                    else:
                        continue

                    #samples_in_rec.append(dataReader(tmp_path))
                    _,data = dataReader(tmp_path)
                    
                    if not os.path.exists(output):
                            os.makedirs(output)

                    with open(output + '/'+tmp_path.split('/')[-1].replace('npz','csv'),'w') as f:
                        for i,feature_vector in enumerate(data):
                                for fet in feature_vector:
                                    f.write("%s " % fet)
                                f.write('\n')
                        f.close 
                    np.save(output + '/'+tmp_path.split('/')[-1].replace('.npz',''), data)

            
def CreateDataset_FinalScorePrediction(original_datapath,h,stride,smoothing,padding):
    for fold in range(1):
        samples_in_rec = []
        fnames = []
        
        train_datapath = original_datapath+'/fold_'+str(fold)+'/train' 
        test_datapath = original_datapath +'/fold_'+str(fold)+'/test' 

        train_success_samples = os.listdir(train_datapath+"/success") 
        train_fail_samples = os.listdir(train_datapath+"/fail") 

        test_success_samples = os.listdir(test_datapath+"/success") 
        test_fail_samples = os.listdir(test_datapath+"/fail") 

        output_datapath = '../../EEG_DATA_DEEP/ScorePrediction/'

        for user in tqdm(range(users)):
            for session in range(mx_num_of_sessions):
                for trial in range(rounds):
                    filename = ('_').join((str(user+1),str(session+1),str(trial+1))) 
                    fnames.append(filename)
                    
                    if filename+'.npz' in train_success_samples:
                        win = '1'
                        output = output_datapath +'user_'+str(user+1)+'/session_'+str(session+1)+'/'+filename+'_'+win
                        tmp_path = ('/').join((train_datapath,"success",filename+'.npz')) 
                   
                    elif filename+'.npz' in train_fail_samples:
                        win = '0'
                        output = output_datapath +'user_'+str(user+1)+'/session_'+str(session+1)+'/'+filename+'_'+win
                        tmp_path = ('/').join((train_datapath,"fail",filename+'.npz')) 
                   
                    elif filename+'.npz' in test_success_samples:
                        win = '1'
                        output = output_datapath +'user_'+str(user+1)+'/session_'+str(session+1)+'/'+filename+'_'+win
                        tmp_path = ('/').join((test_datapath,"success",filename+'.npz')) 
                
                    elif filename+'.npz' in test_fail_samples:
                        win = '0'
                        output = output_datapath +'user_'+str(user+1)+'/session_'+str(session+1)+'/'+filename+'_'+win
                        tmp_path = ('/').join((test_datapath,"fail",filename+'.npz'))
                    else:
                        continue 

                    if not os.path.exists(output):
                            os.makedirs(output)

                    _,data = dataReader(tmp_path)
                    samples = dataSpliter(data,h,stride,smoothing,padding)


                    for i,sample in enumerate(samples):
                        with open(output_datapath + 'user_'+str(user+1)+'/session_'+str(session+1)+'/'+filename+'_'+win+'/'+str(i)+'.csv' ,'w') as f:
                             for feature_vector in sample:
                                for fet in feature_vector:
                                    f.write("%s " % fet)
                                f.write('\n')
                        f.close 
                        np.save(output_datapath + 'user_'+str(user+1)+'/session_'+str(session+1)+'/'+filename+'_'+win+'/'+str(i), sample)

def dataReader(input):
     x = np.load(input)
     
     #total number of sumples is equal to the minimum number of recordings acrosss all signals
     total_samples_in_file = x[x.files[0]].shape[0]
     for i in x.files:
        if i not in ['raw','c','h','h_eeg']:
         #print x[i].shape
         if x[i].shape[0] < total_samples_in_file:
            total_samples_in_file = x[i].shape[0]

    #total_samples_in_file = x['a'].shape[0]
     recording = [] #contains all feature vectors for a single trial

     #[a,b,g,d,t,Aa,Ab,Ag,Ad,At,ascore,bscore,gscore,dscore,tscore] --> 15*4 = 60 features
     for i in range(total_samples_in_file):  
             feature_vector = list(x['a'][i]) + list(x['b'][i]) + list(x['g'][i]) + list(x['d'][i]) + list(x['t'][i]) + list(x['Aa'][i]) + list(x['Ab'][i]) + list(x['Ag'][i]) + list(x['Ad'][i]) + list(x['At'][i]) + list(x['ascore'][i]) + list(x['bscore'][i]) + list(x['gscore'][i]) + list(x['dscore'][i]) + list(x['tscore'][i]) 
             recording.append(feature_vector)
  
     return total_samples_in_file,recording



def dataSpliter(data,HEIGHT,WINDOW_STRIDE,WINDOW_SMOOTHING,PADDING):
    #HEIGHT = 50
    #WINDOW_STRIDE = 5
    #WINDOW_SMOOTHING = 3 # smoothing based on the last 3 samples --> 10Hz sampling rate ==> smoothing based on the last WINDOW_SMOOTHING/10 seconds

    if len(data) < HEIGHT:
        missing_samples = HEIGHT - len(data)
    else:
        missing_samples = WINDOW_STRIDE - len(data)%WINDOW_STRIDE
   
    #print len(data),
    for i in range(missing_samples):
        if PADDING == 0:
        #Augment data with zerro padding
            missing_vector = [0]*len(data[0])
        elif PADDING == 1:
        #Augment data with min value of of past WINDOW_SMOOTHING samples (0.3sec)
            smoothing_sample = np.array(data[-WINDOW_SMOOTHING:], dtype=np.float64)
            missing_vector = np.mean(smoothing_sample, axis=0)
            
        data.append(missing_vector)
   
    samples_all = []
    #steps = float(len(data)%HEIGHT)/float(WINDOW_STRIDE) + 10*(len(data)/100) #number of samples in recording
    steps = (len(data)-50)/5
    for i in range(0,int(WINDOW_STRIDE*steps+WINDOW_STRIDE),WINDOW_STRIDE):
        samples_all.append(data[i:i+HEIGHT])
    #print len(data),steps,np.sum(samples_all[-1][-1]),missing_samples
    return samples_all



    

if __name__ == '__main__':

    users = 69
    mx_num_of_sessions = 5
    rounds = 25
    samples_in_file = []
    args = parseArguments()
   
    
    if args.experimentType == 0:
        CreateDataset_SvF(args.foldPath)
    else:
        CreateDataset_FinalScorePrediction(args.foldPath,args.imageHeight,args.windowStride,args.smoothingStride,args.padding)

 
