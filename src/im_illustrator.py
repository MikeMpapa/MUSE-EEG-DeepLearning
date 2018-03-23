import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
import argparse
import glob, os



def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f' , '--imPath', required=True, help="path to the root of the images")  
    parser.add_argument('-m' , '--mode',choices=[0,1],type=int,required=True, help="0: display mode, 1: save mode")  
    parser.add_argument('-o' , '--saveDir', default='.',type=str,required=False, help="path to save images")  
    
    args = parser.parse_args()
   
    return args



def np_to_image(dir, output,imList,imPaths):

    if not os.path.exists(output):
            os.makedirs(output)

    for i,im in enumerate(imList):
        #im = im.astype(np.uint8)
        #misc.imsave(output+'/'+imPaths[i]+'.png', im)
       # im.imshow(gradient, aspect='auto', cmap='seismic')
        plt.imsave(output+'/'+imPaths[i]+'.png', im,cmap='PiYG',format='png')



def show_images(images, figName,cols = 1, titles = None):
    print figName
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figName)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        #if image.ndim == 2:
            #plt.gray()
        plt.imshow(image, aspect='auto', cmap='seismic')
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()



def main(dir,mode,output):
  
    imList = []

    
    
    if mode == 0 :
        im_paths = [int(filename.split('/')[-1].split('.')[0]) for filename in glob.glob(dir+'*npy')]
        im_paths = sorted(im_paths)
        for i in im_paths:
            imList.append(np.load(dir+str(i)+'.npy').astype(float))
        show_images(imList, dir.split('/')[-2],5,np.array(im_paths).astype('S'))
    else:
        im_paths = [filename.split('/')[-1].split('.')[0] for filename in glob.glob(dir+'*npy')]
        for i in im_paths:
            imList.append(np.load(dir+str(i)+'.npy').astype(float))
        np_to_image(dir, output,imList,np.array(im_paths).astype('S'))



if __name__ == '__main__':
    args = parseArguments()
    main(args.imPath, args.mode, args.saveDir)

