
import pickle
import os
import time

def serialize(var, filename):

    filename=filename+str(hash(time.time()))
    with open(filename, 'wb') as f:            
        pickle.dump(var, f)

    return filename

def deserialize(filename):

    with open(filename, 'rb') as f:            
        return pickle.load(f)

    
