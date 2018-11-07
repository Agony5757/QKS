from pyqpanda import *
from pyqpanda.utils import *
import numpy as np
import math
from numpy.random import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from kitchen_sinks import success_rate,range_to_xy,generate_data,make_dataset

def toy_example_dataset_preparation(training_size=800, test_size=400,
    large_size=2,
    small_size=1,
    large_spread=0.1,
    small_spread=0.05
    ):

    x_data1, y_data1=generate_data(training_size, large_spread, large_size)
    x_data2, y_data2=generate_data(training_size, small_spread, small_size)

    dataset=make_dataset(x_data1,y_data1,x_data2,y_data2)

    x_data_test1, y_data_test1 = generate_data(test_size, large_spread, large_size)
    x_data_test2, y_data_test2 = generate_data(test_size, small_spread, small_size)
    
    testset=make_dataset(x_data_test1,y_data_test1,x_data_test2,y_data_test2)

    return dataset, testset

def toy_example_linear_baseline(dataset, testset, C=1e5):

    X=dataset[0:2,:]
    Y=dataset[2:3,:].flatten()

    logreg = linear_model.LogisticRegression(C=C)
    logreg.fit(X.T, Y)

    test_X=testset[0:2,:]
    test_Y=testset[2:3,:].flatten()
    predict_Y=logreg.predict(test_X.T)

    success, fail = success_rate(predict_Y, test_Y)
    return success/(success+fail)
