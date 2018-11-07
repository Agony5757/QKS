from kitchen_sinks_new import *

from timer import timer

if __name__=='__main__':

    timer.init()
    dataset, testset=toy_example_dataset_preparation()
    
    timer.print_elapse("dataset preparation")
    
    training_feature=dataset[0:2, :]
    training_label = dataset[2, :]

    test_feature=testset[0:2, :]
    test_label = testset[2, :]

    linear_baseline(training_feature.T,training_label,test_feature.T,test_label)

    omega,beta,training_circuit_parameter,test_circuit_parameter=\
        preprocessing(training_feature,test_feature,r=1,force_diagonal=True,E=1000,q=2)
    
    timer.print_elapse("preprocessing")

    raw_train_result,raw_test_result=circuit_run(training_circuit_parameter, 
                                                 test_circuit_parameter, 
                                                 select_ansatz=1)

    timer.print_elapse("circuit run")
    
    raw_train_result=postprocessing(raw_train_result)
    raw_test_result=postprocessing(raw_test_result)

    timer.print_elapse("postprocessing")

    model = training(raw_train_result, training_label)

    timer.print_elapse("train")

    success_rate = testing(model, raw_test_result, test_label)

    timer.print_elapse("test")

    print("QKS Success rate: {}".format(success_rate))
