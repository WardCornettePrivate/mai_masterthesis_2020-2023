import numpy as np
import sklearn

def recursive_len(x, dim=0):
    try:
        dim = recursive_len(x[0], dim+1)
    except:
        pass
    return dim
    
def bags2instances(bags):
    """
    Flattens all bags. Results in one list with all instances of all bags
    """
    return np.array([instance for bag in bags for instance in bag])
