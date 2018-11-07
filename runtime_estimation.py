from pyqpanda import *
from pyqpanda.utils import *

def estimate_runtime(qprog,shots=1):
    for i in range(100):
        result=run_with_configuration(qprog,shots)

