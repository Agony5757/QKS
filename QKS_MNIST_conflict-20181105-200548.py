from MNIST import MyMNIST
import numpy as np
import matplotlib.pyplot as plt

from kitchen_sinks_new import *
from timer import timer

timer.init()
train_img, train_lbl, test_img, test_lbl = MyMNIST.pick([3,5],dataset=MyMNIST.sample(train_sample=1000,test_sample=300))

timer.print_elapse("dataset preparation")

omega,beta,training_circuit_parameter,test_circuit_parameter=\
    preprocessing(train_img.T, test_img.T,q=4,r=196,E=100000)

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