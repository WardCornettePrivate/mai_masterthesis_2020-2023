import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from utils.utils import bags2instances, recursive_len
    
class StandarizerBagsList(BaseEstimator, ClassifierMixin):
    
    def check_exceptions(self, bags):
        if recursive_len(bags) < 3:
           raise Exception('Dimensionerror: Recursive depth of bags = {}'.format(recursive_len(bags)))  


    def fit(self, bags, y=None):
        """ Set standardazing parameters (mean and stdev) based on training population.

        For boolean features (only values 0 and 1) the mean and stdev are set to 0 and 1 respectively

        Parameters
        ----------
        bags : np array of all bags in trainingsset. Dimensions [bag, instance, feature]

        Returns
        -------
        self

        """
        
        # Test if data set has at least three dimensions [bag, instance, feature]
        self.check_exceptions(bags)

        ins = bags2instances(bags)
        
        self.mean = np.mean(ins, axis=0)
        self.std = np.std(ins, axis=0)

        # Loop over binary cols and reset mean and stdev
        for feature in range(0, len(self.mean)):
            values =  np.unique(ins[:, feature])
            
            if np.array_equal(values, np.array([0, 1])):
                self.mean[feature] = 0
                self.std[feature] = 1

        return self
        
    def transform(self, bags):
        """ Transform all instances of in a set of bags to standardized instances

        Parameters
        ----------
        bags : np array bas containg all instances. Dimensions [bag, instance, feature]

        Returns
        -------
        All bags with standardized instances

        """

        # Test if data set has at least three dimensions [bag, instance, feature]
        self.check_exceptions(bags)

        return [[(instance - self.mean) / self.std for instance in bag] for bag in bags]