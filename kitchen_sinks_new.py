from toy_example import (toy_example_dataset_preparation,
                         toy_example_linear_baseline)

import numpy as np
import math
from pyqpanda import *
from pyqpanda.utils import *
from sklearn import linear_model
from timer import timer

import progress_bar

def preprocessing(training_feature, test_feature, **kwargs):
    '''
    N data (dim p)

    p-dim -> q-dim circuit parameter

    Option:
        w_distribution : lambda size (the distribution used to generate w)
        b_distribution : lambda size (the distribution used to generate b)
        q : int (number of circuit parameters)
        E : number of episodes
        r : number of non-zeros
        force_diagonal: boolean (if q==p && r==1 then you can choose to generate a diagonal omega)

    Returns:
        (omega, beta, training_circuit_parameter, test_circuit_parameter)
        omega: the generated omega in the preprocessing
        beta:  the genetared beta in the preprocessing
        training_circuit_parameter: training circuit parameter (np.array with shape (q,N_train))
        test_circuit_parameter: test circuit parameter (np.array with shape(q,N_test))
    '''
    # get shape of features
    training_feature_shape=training_feature.shape
    training_feature_dim=training_feature_shape[0]  # dimension of a feature
    training_feature_size=training_feature_shape[1] # how many features

    test_feature_shape=test_feature.shape
    test_feature_dim=test_feature_shape[0]  # dimension of a feature
    test_feature_size=test_feature_shape[1] # how many features

    assert(test_feature_dim == training_feature_dim)

    feature_dim=test_feature_dim

    # make w and b generator
    w_distribution=lambda size: np.random.normal(scale=2,size=size)
    b_distribution=lambda size: np.random.uniform(low=0,high=2*math.pi,size=size)

    if "w_distribution" in kwargs:
        w_distribution=kwargs["w_distribution"]

    if "b_distribution" in kwargs:
        b_distribution=kwargs["b_distribution"]
    
    # get experiment size (q, E, r)
    q = 2
    E = 1000
    r = 1

    if "q" in kwargs:
        q=kwargs["q"]

    if "E" in kwargs:
        E=kwargs["E"]

    if "r" in kwargs:
        r=kwargs["r"]
    
    # beta generator
    def beta_generator():
        return b_distribution(size=(q,E))

    # omega generator
    def omega_generator():
        omega=np.zeros(shape=(q,feature_dim,E))

        force_diagonal=False
        if "force_diagonal" in kwargs:
            force_diagonal=kwargs["force_diagonal"]

        if r==1 and q==feature_dim and force_diagonal==True:
            raw_omega=w_distribution(size=(q,E))
            for i in range(E):
                for k in range(q):
                    omega[k,k,i]=raw_omega[k,i]

        else:
            def generate_omega_e():
                def generate_omega_row():      
                    omega_non_zeros=w_distribution(size=(r,1))
                    omega_zeros=np.zeros((feature_dim-r,1))
                    omega_=np.vstack((omega_non_zeros,omega_zeros))
                    np.random.shuffle(omega_)
                    return omega_
                omega_e=generate_omega_row()
                for _ in range(q-1):
                    omega_e=np.hstack((omega_e,generate_omega_row()))
                return omega_e.T

            for i in range(E):
                omega[:,:,i]=generate_omega_e()

        return omega

    # preprocessing  
    omega=omega_generator()
    beta=beta_generator()

    # make circuit parameters


    training_circuit_parameter=np.empty((q,E,training_feature_size))

    bar = progress_bar.ProgressBar()
    bar.set_prefix('4/7 preprocess train')
    for i in range(training_feature_size):
        bar.log(i/training_feature_size)
        for e in range(E):
            
            training_circuit_parameter[:,e,i]=beta[:,e]+(omega[:,:,e]).dot(training_feature[:,i])      

    test_circuit_parameter=np.empty((q,E,test_feature_size))

    bar.set_prefix('5/7 preprocess train')
    for i in range(test_feature_size):
        bar.log(i/test_feature_size)
        for e in range(E):
            test_circuit_parameter[:,e,i]=beta[:,e]+(omega[:,:,e]).dot(test_feature[:,i]) 

    return omega, beta, training_circuit_parameter, test_circuit_parameter

def circuit_ansatz_1(qubits, cbits, circuit_parameter):
    '''
    qubit: 2
    cbit: 2
    circuit parameter: 2
    '''
    assert(len(qubits)==2)
    assert(len(cbits)==2)
    assert(len(circuit_parameter)==2)
    prog=QProg()
    prog.insert(RX(qubits[0],circuit_parameter[0]))\
        .insert(RX(qubits[1],circuit_parameter[1]))\
        .insert(meas_all(qubits,cbits))

    return prog

def circuit_ansatz_2(qubits, cbits, circuit_parameter):
    '''
    qubit: 2
    cbit: 2
    circuit parameter: 2
    '''
    assert(len(qubits)==2)
    assert(len(cbits)==2)
    assert(len(circuit_parameter)==2)
    prog=QProg()
    prog.insert(RX(qubits[0],circuit_parameter[0]))\
        .insert(RX(qubits[1],circuit_parameter[1]))\
        .insert(CNOT(qubits[0],qubits[1]))\
        .insert(meas_all(qubits,cbits))

    return prog

def circuit_ansatz_3(qubits, cbits, circuit_parameter):
    '''
    qubit: 4
    cbit: 4
    circuit parameter: 4
    '''
    assert(len(qubits)==4)
    assert(len(cbits)==4)
    assert(len(circuit_parameter)==4)
    prog=QProg()
    for i in range(4):
        prog.insert(RX(qubits[i],circuit_parameter[i]))

    prog.insert(meas_all(qubits,cbits))

    return prog

def circuit_ansatz_4(qubits, cbits, circuit_parameter):
    '''
    qubit: 4
    cbit: 4
    circuit parameter: 4
    '''
    assert(len(qubits)==4)
    assert(len(cbits)==4)
    assert(len(circuit_parameter)==4)
    prog=QProg()
    for i in range(4):
        prog.insert(RX(qubits[i],circuit_parameter[i]))

    prog.insert(CNOT(qubits[0],qubits[1]))\
        .insert(CNOT(qubits[2],qubits[3]))\
        .insert(CNOT(qubits[0],qubits[2]))\
        .insert(CNOT(qubits[1],qubits[3]))

    prog.insert(meas_all(qubits,cbits))

    return prog

def circuit_run(training_parameters, test_parameters, select_ansatz: int, **kwargs):
    n_qubit=None
    n_param=None
    n_cbits=None
    ansatz=None
    if select_ansatz==1:
        n_qubit=2
        n_param=2
        n_cbits=2
        ansatz=circuit_ansatz_1
    elif select_ansatz==2:
        n_qubit=2
        n_param=2
        n_cbits=2
        ansatz=circuit_ansatz_2
    elif select_ansatz==3:
        n_qubit=4
        n_param=4
        n_cbits=4
        ansatz=circuit_ansatz_3
    elif select_ansatz==4:
        n_qubit=4
        n_param=4
        n_cbits=4
        ansatz=circuit_ansatz_4

    training_parameters_shape=training_parameters.shape
    test_parameters_shape=test_parameters.shape
    # q*E*N
    # q - number of parameters
    # E - number of episodes
    # N - number of samples
    
    assert(training_parameters_shape[1]==test_parameters_shape[1])
    assert(training_parameters_shape[0]==test_parameters_shape[0])
    n_train=training_parameters_shape[2]
    n_test=test_parameters_shape[2]
    n_episode=training_parameters_shape[1]
    n_parameters=training_parameters_shape[0]

    assert(n_param==n_parameters)

    raw_train_result=np.empty((n_cbits,n_episode,n_train))
    raw_test_result=np.empty((n_cbits,n_episode,n_test))

    init()
    qubits=qAlloc_many(n_qubit)
    cbits=cAlloc_many(n_cbits)

    bar = progress_bar.ProgressBar()
    bar.set_prefix('6/7 circuit run, train')
    for i in range(n_train):
        for e in range(n_episode):
            bar.log( (e+ i*n_episode) / n_train / n_episode )
            prog=ansatz(qubits,cbits,training_parameters[:,e,i])
            result=run_with_configuration(program=prog,shots=1,cbit_list=cbits)

            for _key in result:
                raw_train_result[:,e,i]=np.array([int(_char) for _char in _key])

    bar.set_prefix('7/7 circuit run, test')
    for i in range(n_test):
        
        for e in range(n_episode):
            bar.log( (e+ i*n_episode) / n_test / n_episode )
            prog=ansatz(qubits,cbits,test_parameters[:,e,i])
            result=run_with_configuration(program=prog,shots=1,cbit_list=cbits)

            for _key in result:
                raw_test_result[:,e,i]=np.array([int(_char) for _char in _key])
    
    finalize()
    return raw_train_result, raw_test_result

def postprocessing(raw_data):
    '''
    raw_data.shape=(q,E,N)
    '''
    return np.array(
            [np.reshape(raw_data[:,:,i],
                (raw_data.shape[1]*raw_data.shape[0])
            ) for i in range(raw_data.shape[2])]
           )

def training(training_data, training_label, C=1e5):
    logreg = linear_model.LogisticRegression(C=C)
    logreg.fit(training_data, training_label)
    return logreg

def success_rate(predict_Y, test_Y):
    success=0
    fail=0
    for i in range(len(predict_Y)):
        if (predict_Y[i] == test_Y[i]):
            success+=1
        else:
            fail+=1
    
    return success,fail

def testing(model, test_data, test_label):
    predict_label = model.predict(test_data)
    success, fail = success_rate(predict_label, test_label)
    return success/(success+fail)

def linear_baseline(train_img, train_lbl, test_img, test_lbl):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_img, train_lbl)

    timer.print_elapse("LR training")

    predict_lbl = logreg.predict(test_img)

    timer.print_elapse("LR predicting")

    success, fail = success_rate(predict_lbl, test_lbl)
    print('LR Linear Baseline: {}'.format(success/(success+fail)))
