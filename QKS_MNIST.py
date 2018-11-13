from MNIST import MyMNIST, _standardize
import numpy as np
import matplotlib.pyplot as plt

from kitchen_sinks_new import *
from timer import timer
from progress_bar import ProgressBar
from ProgressBarGUI import Progress


def preprocessing_mnist(training_feature, test_feature, **kwargs):
    '''
    use split_tile to preprocess the data
    '''
    n_train_samples = training_feature.shape[0]
    n_test_samples  = test_feature.shape[0]

    #print('{} train samples'.format(n_train_samples))
    #print('{} test samples'.format(n_test_samples))

    rawdata=training_feature[0]
    train_tiles = split_tile(rawdata)
    tile_shape = train_tiles.shape
    # such as
    # 20000* 784  would be split into
    # 196 * 2 * 2 * 20000
    tile_len = tile_shape[0]  # 196
    tile_n = tile_shape[1]    # 2

    train_tiles = np.empty((tile_len, tile_n, tile_n, n_train_samples))
    test_tiles = np.empty((tile_len, tile_n, tile_n, n_test_samples))

    bar = Progress('train split')
    for i in range(n_train_samples):
        rawdata=training_feature[i] # (784,)
        split_tiles = split_tile(rawdata) #(196,2,2)
        bar.log(i/n_train_samples)
        for n in range(tile_n):
            for m in range(tile_n):
                train_tiles[:,n,m,i]=split_tiles[:,n,m]
    bar.destroy()

    bar = Progress('test split')
    for i in range(n_test_samples):
        rawdata=test_feature[i]
        split_tiles = split_tile(rawdata)
        bar.log(i/n_test_samples)
        for n in range(tile_n):
            for m in range(tile_n):
                test_tiles[:,n,m,i]=split_tiles[:,n,m]
    bar.destroy()
    
    E=1000
    if 'E' in kwargs:
        E=math.ceil(kwargs['E'])

    w_distribution=lambda size: np.random.normal(scale=2,size=size)
    b_distribution=lambda size: np.random.uniform(low=0,high=2*math.pi,size=size)

    if "w_distribution" in kwargs:
        w_distribution=kwargs["w_distribution"]

    if "b_distribution" in kwargs:
        b_distribution=kwargs["b_distribution"]

    train_circuit_parameter = np.empty((tile_n*tile_n, E, n_train_samples))
    test_circuit_parameter  = np.empty((tile_n*tile_n, E, n_test_samples))    

    for i in range(tile_n):
        for j in range(tile_n):
            _,_,train_prmt,test_prmt=preprocessing(train_tiles[:,i,j,:],test_tiles[:,i,j,:], 
                                                    q=1, E=E, r=tile_len,
                                                    w_distribution= w_distribution,
                                                    b_distribution= b_distribution)
            train_circuit_parameter[i*tile_n+j,:,:]=train_prmt
            test_circuit_parameter[i*tile_n+j,:,:]=test_prmt

    return None,None,train_circuit_parameter,test_circuit_parameter    

def split_tile(rawdata, split=2):
    '''
    data should be in 784*1

    Returns:
        tiles: np.array with shape (size, n_split, n_split)
    '''

    # first reshape the raw data into a 28*28 picture

    picture=np.reshape(rawdata,(28,28))
    
    # split 28
    if 28 % split == 0:
        each = 28 // split
    else:
        each = 28 // split + 1
    
    if each<=0:
        # split>=28 then each data is splitted
        tiles=np.empty((1,28,28))
        for i in range(28):
            for j in range(28):
                tiles[0,i,j]=picture[i,j]

        return tiles

    else:
        tiles=np.empty((each,each,split,split))

        for i in range(split):
            for j in range(split):
                for k in range(each):
                    for l in range(each):
                        row = i*each+k
                        col = j*each+l
                        if row<28 and col<28:
                            tiles[k,l,i,j]=picture[row,col]

        return_tiles = np.empty((each*each, split, split))
        for i in range(split):
            for j in range(split):
                for k in range(each):
                    for l in range(each):
                        return_tiles[:,i,j]=np.reshape(tiles[:,:,i,j],(each*each))

        return return_tiles

def show_tiles():
    train_img, train_lbl, test_img, test_lbl = MyMNIST.pick([3,5],dataset=MyMNIST.sample(train_sample=1000,test_sample=300))

    img=train_img[0]
    img=np.reshape(img,(28,28))
    fig = plt.figure(1)
    plt.imshow(img,cmap=plt.get_cmap('gray_r'))

    tiles=split_tile(img)
    fig=plt.figure(2)
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)
    tiles00 = np.reshape(tiles[:,0,0],(14,14))
    tiles01 = np.reshape(tiles[:,0,1],(14,14))
    tiles10 = np.reshape(tiles[:,1,0],(14,14))
    tiles11 = np.reshape(tiles[:,1,1],(14,14))

    ax1.imshow(tiles00,cmap=plt.get_cmap('gray_r'))
    ax2.imshow(tiles01,cmap=plt.get_cmap('gray_r'))
    ax3.imshow(tiles10,cmap=plt.get_cmap('gray_r'))
    ax4.imshow(tiles11,cmap=plt.get_cmap('gray_r'))
    plt.show()

def naive_implementation():
    timer.init()
    train_img, train_lbl, test_img, test_lbl = MyMNIST.pick([3,5],dataset=MyMNIST.sample(train_sample=1000,test_sample=300))

    timer.print_elapse("dataset preparation")

    omega,beta,training_circuit_parameter,test_circuit_parameter=\
        preprocessing(train_img.T, test_img.T,q=4,r=196,E=10000)

    timer.print_elapse("preprocessing")

    raw_train_result,raw_test_result=circuit_run(training_circuit_parameter, 
                                                test_circuit_parameter, 
                                                select_ansatz=4)

    timer.print_elapse("circuit run")
        
    raw_train_result=postprocessing(raw_train_result)
    raw_test_result=postprocessing(raw_test_result)

    timer.print_elapse("postprocessing")

    model = training(raw_train_result, train_lbl)

    timer.print_elapse("train")

    success_rate = testing(model, raw_test_result, test_lbl)

    timer.print_elapse("test")

    print("QKS Success rate: {}".format(success_rate))

def tile_implementation(**kwargs):

    '''
    ALL OPTIONS:
        1. pick (default: [3,5])
        2. sample (default: True)
        3. (sample==True) train_sample (default: 3000)
        4. (sample==True) test_sample  (default: 1000)
        5. standardize (default: 'normalize')
        6. w_dist_type (default: 'normal')
        7. b_dist_type (default: 'uniform')
        8. w_dist_param (default: (0, 0.5))
        9. b_dist_param (default: (0, math.pi))
        10. E (default: 2000)

    RETURNS:
        1. Accuracy
        2. Model
    '''

    timer.init()

    pick_range = None
    if 'pick' in kwargs:
        pick_range = kwargs['pick']
    else:
        pick_range = [3,5]

    is_sample = None
    train_sample = None
    test_sample = None

    if 'sample' in kwargs:
        is_sample = kwargs['sample']
        if is_sample is True:
            train_sample = kwargs['train_sample']
            test_sample = kwargs['test_sample']        
    else:
        is_sample = True
        train_sample = 3000
        test_sample = 1000

    if is_sample:
        train_img, train_lbl, test_img, test_lbl = \
            MyMNIST.pick(pick_range ,dataset=MyMNIST.sample(train_sample=train_sample,test_sample=test_sample))
    else:
        train_img, train_lbl, test_img, test_lbl = MyMNIST.pick(pick_range)

    standardize = None 
    if 'standardize' in kwargs:
        standardize = kwargs['standardize']
    else:
        standardize = 'normalize'

    train_img = _standardize(train_img, standardize)
    test_img  = _standardize(test_img, standardize)

    w_distribution = None
    b_distribution = None
    w_dist_type = None
    b_dist_type = None
    w_dist_param = None
    b_dist_param = None

    if 'w_dist_type' in kwargs:
        w_dist_type = kwargs['w_dist_type']
    else:
        w_dist_type = 'normal'

    if 'w_dist_param' in kwargs:
        w_dist_param = kwargs['w_dist_param']
    else:
        w_dist_param = (0,0.5)

    if 'b_dist_type' in kwargs:
        b_dist_type = kwargs['b_dist_type']
    else:
        b_dist_type = 'uniform'

    if 'b_dist_param' in kwargs:
        b_dist_param = kwargs['b_dist_param']
    else:
        b_dist_param = (0,math.pi)

    if w_dist_type == 'normal':
        w_distribution = lambda size : np.random.normal(w_dist_param[0], w_dist_param[1], size)
    elif w_dist_type == 'uniform':
        w_distribution = lambda size : np.random.uniform(w_dist_param[0], w_dist_param[1], size)

    if b_dist_type == 'normal':
        b_distribution = lambda size : np.random.normal(b_dist_param[0], b_dist_param[1], size)
    elif b_dist_type == 'uniform':
        b_distribution = lambda size : np.random.uniform(b_dist_param[0], b_dist_param[1], size)

    E = 2000
    if 'E' in kwargs:
        E=kwargs['E']

    _,_,training_circuit_parameter, test_circuit_parameter = preprocessing_mnist(train_img, test_img, E=E,
                                                                            w_distribution= w_distribution,
                                                                            b_distribution= b_distribution)                                                                            

    #timer.print_elapse("QKS preprocessing")

    raw_train_result,raw_test_result=circuit_run(training_circuit_parameter, 
                                                 test_circuit_parameter, 
                                                 select_ansatz=4)

    #timer.print_elapse("QKS circuit run")
    
    raw_train_result=postprocessing(raw_train_result)
    raw_test_result=postprocessing(raw_test_result)

    #timer.print_elapse("QKS postprocessing")

    model = training(raw_train_result, train_lbl)

    #timer.print_elapse("QKS train")

    success_rate = testing(model, raw_test_result, test_lbl)

    #timer.print_elapse("QKS test")

    #print("QKS Success rate: {}".format(success_rate))

    return success_rate, model

from logger import *
from ProgressBarGUI import *

def scan_E_scale(E_range, scale_range):

    logger.init(foldername='scan E-Scale',prefix='Scan2D')
    logger.title('E, Scale, SuccessRate')    

    #bar = Progress('Main Scan',on=True)
    bar = ProgressBar()
    
    all=len(E_range)*len(scale_range)
    i=1

    for scale in scale_range:
        for E in E_range:       
            
            bar.log( i/all )
            success_rate = tile_implementation(pick=[3,5],
                                sample=True,
                                train_sample=3000,
                                test_sample=1000,
                                standardize='normalize',
                                w_dist_type='normal',
                                w_dist_param=(0, scale),
                                b_dist_type= 'uniform',
                                b_dist_param=(0, math.pi),
                                E = E
                            )
            i+=1
            data = (E, scale, success_rate)
            logger.log(data)

    logger.finalize()

from pickle import dump

if __name__=='__main__':
    '''
    E_range = np.linspace(100,5000,num=20)
    scale_range = np.linspace(0.01, 1, num=20)
    scan_E_scale(E_range= E_range, scale_range=scale_range)
    '''

    pick_dataset = [0,1,2,3,4,5,6,7,8,9]

    scale = 0.25
    Episode = 5000
    filename = 'SCALE {} EPISODE {}.model.qks'.format(scale,Episode)
    print(filename)
    success_rate, model = tile_implementation(pick=pick_dataset,
                                sample=False,
                                standardize='normalize',
                                w_dist_type='normal',
                                w_dist_param=(0, scale),
                                b_dist_type= 'uniform',
                                b_dist_param=(0, math.pi),
                                E = Episode
                            )

    
    with open(filename,'wb') as fp:
        dump(model, fp)

    print('Digits:', pick_dataset)
    print('Accuracy:',success_rate)