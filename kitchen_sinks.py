from pyqpanda import *
from pyqpanda.utils import *
import numpy as np
import math
from numpy.random import *
import matplotlib.pyplot as plt
from sklearn import linear_model

def qkernel_2_2_2__1(q, theta, c, _assert=True):
    '''
    q: dim-2 qubit vector\n
    theta: dim-2 float\n
    c: dim-2 cbit vector
    '''
    if _assert is True:
        assert(len(q)==2)
        assert(len(theta)==2)
        assert(len(c)==2)

    prog=QProg()                   \
        .insert(RX(q[0],theta[0])) \
        .insert(RX(q[1],theta[1])) \
        .insert(meas_all(q,c))

    return prog

def qkernel_2_2_2__2(q, theta, c, _assert=True):
    '''
    q: dim-2 qubit vector\n
    theta: dim-2 float\n
    c: dim-2 cbit vector    
    '''
    if _assert is True:
        assert(len(q)==2)
        assert(len(theta)==2)
        assert(len(c)==2)

    prog=QProg()
    prog.insert(single_gate_apply_to_all(gate=H,qubit_list=q))\
        .insert(CZ(q[0],q[1]))  \
        .insert(RX(q[0],theta[0]))  \
        .insert(RX(q[1],theta[1]))  \
        .insert(meas_all(q,c))

    return prog

def qkernel_4_4_4__1(qubit, theta, c, _assert=True):

    '''
    q: dim-4 qubit vector\n
    theta: dim-4 float\n
    c: dim-4 cbit vector    
    '''
    
    if _assert is True:
        assert(len(qubit)==4)
        assert(len(theta)==4)
        assert(len(c)==4)

    prog=QProg()
    prog.insert(RX(qubit[0],theta[0]))  \
        .insert(RX(qubit[2],theta[2]))  \
        .insert(RX(qubit[3],theta[3]))  \
        .insert(RX(qubit[1],theta[1]))  \
        .insert(CNOT(qubit[0],qubit[1]))\
        .insert(CNOT(qubit[2],qubit[3]))\
        .insert(CNOT(qubit[0],qubit[2]))\
        .insert(CNOT(qubit[1],qubit[3]))\
        .insert(meas_all(qubit,c))

    return prog    
    
class quantum_kitchen_sinks:
        
    '''
    Page3
    ...under the LB rule, we require that the mapping from data to angles be 
    linear. To define a linear encoding, let Ui belongs to R^p for i=1..M be
    a p-dim input vector from a dataset containing M examples.

    We can encode this input vector into q gate parameters using a (q*p)-dim
    matrix Oe of the form Oe=(w1,..wq)^T where wk is a p-dim vector with a 
    number r<=p elements being random values and the other elements being exact
    zero.
    '''

    '''
    dataset should be alike : 
    
        shape= (n_examples, n_features)    
        data=
        [[1,2],
         [2,3],
         [3,4],
         [4,5],
         ...]
         
    '''
    
    def __init__(self, 
                dataset, 
                n_parameters_q, 
                n_episode,
                non_zero_terms_r,
                omega_distribution=lambda size: np.random.normal(scale=2,size=size),
                beta_distribution=lambda size: np.random.uniform(low=0,high=2*math.pi,size=size)
    ):
        self.dim_p=len(dataset[:,0])
        self.n_example=len(dataset[0,:])
        self.dataset=dataset
        self.n_parameters_q=n_parameters_q
        self.n_episode=n_episode
        self.omega=list()
        self.beta=list()
        self.non_zero_terms_r=non_zero_terms_r
        self.omega_distribution=omega_distribution
        self.beta_distribution=beta_distribution

        self.generate_omega_and_beta()

    def generate_one_episode_1(self):        
        def generate_omega_row():      
            # generate a r*1 matrix   
            omega_non_zeros=self.omega_distribution(size=(self.non_zero_terms_r,1))
            # generate a (p-r)*1 matrix
            omega_zeros=np.zeros((self.dim_p-self.non_zero_terms_r,1))
            # stack them
            omega_=np.vstack((omega_non_zeros,omega_zeros))
            # shuffle
            np.random.shuffle(omega_)
            return omega_
        
        omega_e=generate_omega_row()
        for _ in range(self.n_parameters_q-1):
            omega_e=np.hstack((omega_e,generate_omega_row()))

        beta_e=self.beta_distribution(size=(self.n_parameters_q,1))

        return omega_e.T, beta_e

    def generate_one_episode_2(self):        
        def generate_omega_row():      
            # generate a r*1 matrix   
            omega_non_zeros=self.omega_distribution(size=(self.non_zero_terms_r,1))
            # generate a (p-r)*1 matrix
            omega_zeros=np.zeros((self.dim_p-self.non_zero_terms_r,1))
            # stack them
            omega_=np.vstack((omega_non_zeros,omega_zeros))
            # shuffle
            np.random.shuffle(omega_)
            return omega_
        
        omega_e=generate_omega_row()
        for _ in range(self.n_parameters_q-1):
            omega_e=np.hstack((omega_e,generate_omega_row()))

        beta_e=self.beta_distribution(size=(self.n_parameters_q,1))

        return omega_e.T, beta_e

    def generate_omega_and_beta(self, generate_episode='1'):
        
        if generate_episode=='1':
            for i in range(self.n_episode):
                omega_e,beta_e=self.generate_one_episode_1()
                self.omega.append(omega_e)
                self.beta.append(beta_e)   
        else:
            raise AttributeError 
    
    def _test_is_shape_match(self):
        try:
            omega=self.omega[0]
            beta=self.beta[0]
            inputvector=self.dataset[0]
            print('omega=\n',omega)
            print('beta=\n',beta)
            print('inputvector=\n',inputvector)
            theta=omega.dot(inputvector)+beta
        except:
            print('shape_not_match')
        else:
            print('shape_match')

    def __str__(self):
        ''' ready for print '''
        retstr=''
        retstr+= '* Configuration of Quantum Kitchen Sinks *\n'
        retstr+= 'p= {} (dimension of input vector)\n'.format(self.dim_p)
        retstr+= 'q= {} (number of gate parameters)\n'.format(self.n_parameters_q)
        retstr+= 'M= {} (size of training set)\n'.format(self.n_example)
        retstr+= 'E= {} (number of episodes)\n'.format(self.n_episode)
        retstr+= 'r= {} (number of non-zero terms)\n'.format(self.non_zero_terms_r)
        retstr+= 'Omega Shape: {} Episode= {}\n'.format(self.omega[0].shape, len(self.omega))
        retstr+= 'Beta Shape: {} Episode= {}\n'.format(self.beta[0].shape, len(self.beta))
        retstr+= 'Dataset Shape: {} Number= {}\n'.format(self.dataset.shape, len(self.dataset[0,:]))
                
        return retstr
    
    def get_rotation_angles(self, i_episode, i_data):
        omega_e=self.omega[i_episode]
        beta_e=self.beta[i_episode]
        u_i=self.dataset[:,i_data:i_data+1]
        theta=omega_e.dot(u_i)+beta_e
        return theta.T.tolist()[0]
    
    def set_dataset(self,dataset):
        self.dataset=dataset
        self.n_example=len(self.dataset[0,:])
    
    def run(self, kernel=qkernel_2_2_2__1):         
        init()
        q=qAlloc_many(2)
        c=cAlloc_many(2)
        
        def runkernel(theta):        
            prog=kernel(q,theta,c)
            result=run_with_configuration(program=prog,shots=1,cbit_list=c)
            
            for _key in result:
                # there should be only 1 key with value==1
                
                return [int(_char) for _char in _key]
        
        results=list()
        for i_data in range(self.n_example):
            result=list()
            for i_episode in range(self.n_episode):
                # stack the results
                result+=runkernel(self.get_rotation_angles(i_episode,i_data))
            results.append(result)
        finalize()
        return np.array(results)
    
# DATA GENERATION

def range_to_xy(l,r,e,size,range):
    if e==0:        # left
        x=-size+r
        y=l
    elif e==1:      # upper
        x=l
        y=size+r
    elif e==2:      # right
        x=size+r
        y=l
    elif e==3:      # lower
        x=l
        y=-size+r

    return x,y

def generate_data(data_size, range_, size):
    range_l_1=uniform(low=-size, high=size, size=data_size)
    range_r_1=normal(scale=range_,size=data_size)    
    range_e_1=randint(low=0, high=4, size=data_size)
    x_data=np.zeros(data_size)
    y_data=np.zeros(data_size)
    for i in range(data_size):
        x,y=range_to_xy(range_l_1[i], range_r_1[i], range_e_1[i], size, range_)
        x_data[i]=x
        y_data[i]=y

    return x_data, y_data

def make_dataset(x_data1, y_data1, x_data2, y_data2):
    dataset=np.zeros((3,len(x_data1)+len(x_data2)))
    for i in range(len(x_data1)):
        dataset[0,i]=x_data1[i]
        dataset[1,i]=y_data1[i]
        dataset[2,i]=1
        
    for i in range(len(x_data2)):
        dataset[0,i+len(x_data1)]=x_data2[i]
        dataset[1,i+len(x_data1)]=y_data2[i]
        dataset[2,i+len(x_data1)]=-1
        
    return dataset

def success_rate(predict_Y, test_Y):
    success=0
    fail=0
    for i in range(len(predict_Y)):
        if (predict_Y[i] == test_Y[i]):
            success+=1
        else:
            fail+=1
    
    return success,fail
# logistic regression applied directly on the dataset, and generated the 50% success prediction rate

if __name__=='__main__':

    x_data1,y_data1=generate_data(800,0.1,2)
    x_data2,y_data2=generate_data(800,0.05,1)

    dataset=make_dataset(x_data1,y_data1,x_data2,y_data2)
    x_data_test1, y_data_test1 = generate_data(400,0.1,2)
    x_data_test2, y_data_test2 = generate_data(400,0.05,1)

    testset=make_dataset(x_data_test1,y_data_test1,x_data_test2,y_data_test2)

    X=dataset[0:2,:]
    Y=dataset[2:3,:].flatten()

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X.T, Y)

    test_X=testset[0:2,:]
    test_Y=testset[2:3,:].flatten()
    predict_Y=logreg.predict(test_X.T)

    success, fail = success_rate(predict_Y, test_Y)
    print('WITHOUT QUANTUM RESOURCE: success_rate=', success/(success+fail))

    x_data1,y_data1=generate_data(800,0.1,2)
    x_data2,y_data2=generate_data(800,0.05,1)

    dataset=make_dataset(x_data1,y_data1,x_data2,y_data2)

    x_data_test1, y_data_test1 = generate_data(400,0.1,2)
    x_data_test2, y_data_test2 = generate_data(400,0.05,1)

    testset=make_dataset(x_data_test1,y_data_test1,x_data_test2,y_data_test2)


    # perform on feature of train data
    X=dataset[0:2,:]
    Y=dataset[2:3,:]

    instance=quantum_kitchen_sinks(
                    dataset=X, 
                    n_parameters_q=2, 
                    n_episode=100,
                    non_zero_terms_r=1
    )
    X=instance.run()

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y.T)

    # perform on feature of test data
    test_X=testset[0:2,:]
    test_Y=testset[2:3,:]

    instance.set_dataset(test_X)
    test_X=instance.run()

    predict_Y=logreg.predict(test_X)

    success, fail = success_rate(predict_Y, test_Y)
    print('QUANTUM KITCHEN SINKS: success_rate=', success/(success+fail))