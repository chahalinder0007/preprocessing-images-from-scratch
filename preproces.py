import os
if not os.path.exists("dumps"):
    import depends 
    os.makedirs("dumps")   
    
    
import cv2,math
import numpy as np
from scipy import ndimage
from pywt import dwt
from sklearn.utils import shuffle
from scipy.fftpack import dct


class preprocess():
    def __init__(self,path="SAMPLE/",thresh=200,size=(140,140)):
        assert type(size) is tuple and len(size)==2,"size should be a tuple of len 2:: e.g. (rows,cols)"
        self.path=path
        self.thresh=thresh
        self.size=size
    
    def cutter(self,raw_image):
        while np.sum(raw_image[0]) == 0: #deletes the top row as long as there are no alive pixels
            raw_image = raw_image[1:]
        while np.sum(raw_image[:,0]) == 0: #similarly deletes the bottom row till its sum is zero..and so on
            raw_image = np.delete(raw_image,0,1)
        while np.sum(raw_image[-1]) == 0:
            raw_image = raw_image[:-1]
        while np.sum(raw_image[:,-1]) == 0:
            raw_image = np.delete(raw_image,-1,1)
        rows,cols = raw_image.shape     
        return rows,cols,raw_image   
    def b_shif(self,inpu_image):
        coef_y,coef_x = ndimage.measurements.center_of_mass(inpu_image)
        rows,cols = inpu_image.shape
        shft_x = np.round(cols/2.0-coef_x).astype(int)
        shift_y = np.round(rows/2.0-coef_y).astype(int)
        return shft_x,shift_y
    def shifter(self,inpu_image,s_x,s_y):
        rows,cols = inpu_image.shape
        mid_val = np.float32([[1,0,s_x],[0,1,s_y]])
        shifted = cv2.warpAffine(inpu_image,mid_val,(cols,rows))
        return shifted
    
    def data(self,type_of_transform="dwt and dct",shuffling=True):
        assert type_of_transform=="dwt" or type_of_transform=="dct" or type_of_transform=="dwt and dct","type_of_transform can either be dct(discreate cosine transform) or dwt (discreate wave transform) or dwt and dct"
        X=[]
        Y=[]
        
        if type_of_transform=="dwt":
            for dirs in os.listdir(path=self.path):
                for images in os.listdir(path="%s%s"%(self.path,dirs)):
                    raw_image = cv2.imread("%s/%s/%s"%(self.path,dirs,images), 0)

                    raw_image = cv2.resize(255-raw_image, self.size)
                    (thresh, raw_image) = cv2.threshold(raw_image, self.thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    #raw_raw.point(lambda x:x > thresh and 255)  
                    rows,cols,raw_image=self.cutter(raw_image)
                    if rows > cols:
                        factor = (self.size[1]-10)/rows #changes
                        rows = (self.size[1]-10)
                        cols = int(round(cols*factor))
                        raw_image = cv2.resize(raw_image, (cols,rows))
                    else:
                        factor = (self.size[1]-10)/cols
                        cols = (self.size[1]-10)
                        rows = int(round(rows*factor))
                        raw_image = cv2.resize(raw_image, (cols, rows))


                    c_pad = (int(math.ceil((self.size[1]-cols)/2.0)),int(math.floor((self.size[1]-cols)/2.0)))
                    raw_pad = (int(math.ceil((self.size[0]-rows)/2.0)),int(math.floor((self.size[0]-rows)/2.0)))
                    raw_image = np.lib.pad(raw_image,(raw_pad,c_pad),'constant')
                    shft_x,shift_y = self.b_shif(raw_image)
                    shifted = self.shifter(raw_image,shft_x,shift_y)
                    raw_image = shifted

                    flatten = raw_image.flatten() / 255.0
                    #transform will come here

                    flatten,disc=dwt(flatten,wavelet="db1")
                    X.append(flatten)
                    Y.append(dirs)
            if shuffling:
                X,Y=shuffle(X,Y,random_state=0)
            return X,Y
        
        elif type_of_transform=="dct":
            for dirs in os.listdir(path=self.path):
                for images in os.listdir(path="%s%s"%(self.path,dirs)):
                    raw_image = cv2.imread("%s/%s/%s"%(self.path,dirs,images), 0)

                    raw_image = cv2.resize(255-raw_image, self.size)
                    (thresh, raw_image) = cv2.threshold(raw_image, self.thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    #raw_raw.point(lambda x:x > thresh and 255)  
                    rows,cols,raw_image=self.cutter(raw_image)
                    if rows > cols:
                        factor = (self.size[1]-10)/rows #changes
                        rows = (self.size[1]-10)
                        cols = int(round(cols*factor))
                        raw_image = cv2.resize(raw_image, (cols,rows))
                    else:
                        factor = (self.size[1]-10)/cols
                        cols = (self.size[1]-10)
                        rows = int(round(rows*factor))
                        raw_image = cv2.resize(raw_image, (cols, rows))

                    c_pad = (int(math.ceil((self.size[1]-cols)/2.0)),int(math.floor((self.size[1]-cols)/2.0)))
                    raw_pad = (int(math.ceil((self.size[0]-rows)/2.0)),int(math.floor((self.size[0]-rows)/2.0)))
                    raw_image = np.lib.pad(raw_image,(raw_pad,c_pad),'constant')
                    shft_x,shift_y = self.b_shif(raw_image)
                    shifted = self.shifter(raw_image,shft_x,shift_y)
                    raw_image = shifted

                    flatten = raw_image.flatten() / 255.0
                    #transform will come here

                    flatten=dct(flatten)
                    X.append(flatten)
                    Y.append(dirs)
            if shuffling:
                X,Y=shuffle(X,Y,random_state=0)
            return X,Y  
        elif type_of_transform=="dwt and dct":
            for dirs in os.listdir(path=self.path):
                for images in os.listdir(path="%s%s"%(self.path,dirs)):
                    raw_image = cv2.imread("%s/%s/%s"%(self.path,dirs,images), 0)

                    raw_image = cv2.resize(255-raw_image, self.size)
                    (thresh, raw_image) = cv2.threshold(raw_image, self.thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    #raw_raw.point(lambda x:x > thresh and 255)  
                    rows,cols,raw_image=self.cutter(raw_image)
                    if rows > cols:
                        factor = (self.size[1]-10)/rows #changes
                        rows = (self.size[1]-10)
                        cols = int(round(cols*factor))
                        raw_image = cv2.resize(raw_image, (cols,rows))
                    else:
                        factor = (self.size[1]-10)/cols
                        cols = (self.size[1]-10)
                        rows = int(round(rows*factor))
                        raw_image = cv2.resize(raw_image, (cols, rows))

                    c_pad = (int(math.ceil((self.size[1]-cols)/2.0)),int(math.floor((self.size[1]-cols)/2.0)))
                    raw_pad = (int(math.ceil((self.size[0]-rows)/2.0)),int(math.floor((self.size[0]-rows)/2.0)))
                    raw_image = np.lib.pad(raw_image,(raw_pad,c_pad),'constant')
                    shft_x,shift_y = self.b_shif(raw_image)
                    shifted = self.shifter(raw_image,shft_x,shift_y)
                    raw_image = shifted

                    flatten_m = raw_image.flatten() / 255.0
                    #transform will come here
                    flatten_2,disc=dwt(flatten_m,wavelet="db1")
                    flatten_1=dct(flatten_m)
                    #print(np.shape(flatten_2),np.shape(flatten_1))
                    flatten=np.concatenate((flatten_1,flatten_2))
                    X.append(flatten)
                    Y.append(dirs)
            if shuffling:
                X,Y=shuffle(X,Y,random_state=0)
            return X,Y            