import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import struct
from time import time


train_data  = "train-images.idx3-ubyte"
train_label = "train-labels.idx1-ubyte"

test_data   = "t10k-images.idx3-ubyte"
test_label  = "t10k-labels.idx1-ubyte"

from timer import timer

class MyMNIST:  

    train_img=None
    train_lbl=None
    test_img=None
    test_lbl=None

    train_data=None,
    train_label=None,               
    test_data=None,
    test_label=None

    def __init__(self):
        pass

    @staticmethod
    def load(train_data="train-images.idx3-ubyte",
             train_label="train-labels.idx1-ubyte",               
             test_data="t10k-images.idx3-ubyte",
             test_label="t10k-labels.idx1-ubyte",
             reload=False,
             standardize=True):

        MyMNIST.train_data=train_data
        MyMNIST.train_label=train_label
        MyMNIST.test_data=test_data
        MyMNIST.test_label=test_label

        if (MyMNIST.train_img is None) or\
          (MyMNIST.train_lbl is None) or\
          (MyMNIST.test_img  is None) or\
          (MyMNIST.test_lbl  is None) or\
          (reload is True):
                MyMNIST.train_img=MyMNIST.load_data(train_data)
                MyMNIST.train_lbl=MyMNIST.load_label(train_label)
                MyMNIST.test_img=MyMNIST.load_data(test_data)
                MyMNIST.test_lbl=MyMNIST.load_label(test_label)


        
        if standardize is True:

            def _standardize(img):
                E = np.mean(img, axis=1, keepdims=True)
                Var = np.var(img, axis=1, keepdims=True)

                return (img-E)/Var
            
            MyMNIST.train_img= _standardize(MyMNIST.train_img)
            MyMNIST.test_img = _standardize(MyMNIST.test_img)
        
        return MyMNIST.train_img, MyMNIST.train_lbl, MyMNIST.test_img, MyMNIST.test_lbl

    @staticmethod
    def load_data(filename):
        data_fp= open(filename, 'rb')
        data = data_fp.read()
        head = struct.unpack_from('>IIII',data,0)
        offset = struct.calcsize('>IIII')
        imgNum = head[1]
        width = head[2]
        height = head[3]
        bits = imgNum * width * height
        bitsString = '>' + str(bits) + 'B'
        imgs = struct.unpack_from(bitsString,data,offset)
        data_fp.close()
        im = np.reshape(np.array(imgs),(imgNum,width*height))
        return im

    @staticmethod
    def load_label(filename):
        label_fp = open(filename, 'rb')
        label = label_fp.read() 
        head = struct.unpack_from('>II' , label ,0)
        imgNum=head[1]    
        offset = struct.calcsize('>II')
        numString = '>'+str(imgNum)+"B"
        labels = struct.unpack_from(numString,label,offset)
        label_fp.close()
        labels = np.reshape(labels,(imgNum))
        return labels

    @staticmethod
    def sample(train_sample=1000,test_sample=100, standardize=True):
        MyMNIST.load(standardize=standardize)
        train_range=len(MyMNIST.train_img[:,0])
        train_range=range(train_range)
        train_d=np.random.choice(train_range, size=train_sample)
        train_img=MyMNIST.train_img[train_d]
        train_lbl=MyMNIST.train_lbl[train_d]
        
        test_range=len(MyMNIST.test_img[:,0])
        test_range=range(test_range)
        test_d=np.random.choice(test_range, size=test_sample)
        test_img=MyMNIST.test_img[test_d]
        test_lbl=MyMNIST.test_lbl[test_d]
        
        return (train_img, train_lbl, test_img, test_lbl)

    @staticmethod
    def pick(number_list,
             dataset=None,
             standardize=True):
        MyMNIST.load(standardize=standardize)
        train_img=None
        train_lbl=None
        test_img=None
        test_lbl=None
        if dataset is None:
            train_img=MyMNIST.train_img
            train_lbl=MyMNIST.train_lbl
            test_img =MyMNIST.test_img
            test_lbl =MyMNIST.test_lbl
        else:
            train_img=dataset[0]
            train_lbl=dataset[1]
            test_img =dataset[2]
            test_lbl =dataset[3]

        picked_train_img=list()
        picked_train_lbl=list()
        picked_test_img =list()
        picked_test_lbl =list()
        for i in range(len(train_img)):
            if train_lbl[i] in number_list:
                picked_train_img.append(train_img[i])
                picked_train_lbl.append(train_lbl[i])
        
        picked_train_img=np.array(picked_train_img)
        picked_train_lbl=np.array(picked_train_lbl)

        for i in range(len(test_img)):
            if test_lbl[i] in number_list:
                picked_test_img.append(test_img[i])
                picked_test_lbl.append(test_lbl[i])
        
        picked_test_img=np.array(picked_test_img)
        picked_test_lbl=np.array(picked_test_lbl)
        
        return picked_train_img, picked_train_lbl,picked_test_img,picked_test_lbl


def success_rate(predict_Y, test_Y):
    success=0
    fail=0
    for i in range(len(predict_Y)):
        if (predict_Y[i] == test_Y[i]):
            success+=1
        else:
            fail+=1
    
    return success,fail

if __name__ == "__main__":
    timer.init()

    train_img, train_lbl, test_img, test_lbl = MyMNIST.load()

    timer.print_elapse("load data")
    
    #train_img, train_lbl, test_img, test_lbl = MyMNIST.sample()

    #timer.print_elapse("sample data") 

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_img, train_lbl)

    timer.print_elapse("training")

    predict_lbl = logreg.predict(test_img)

    timer.print_elapse("predicting")

    success, fail = success_rate(predict_lbl, test_lbl)
    print('success_rate=', success/(success+fail))

    timer.end()