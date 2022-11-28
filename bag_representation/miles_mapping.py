import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from utils.utils import recursive_len, bags2instances

class MILESMapping(BaseEstimator, TransformerMixin):
    """
    
    """
    
    def __init__(self, sigma2=4.5**2, return_iip_bag_closest=True, print_progress=True):
        """
        Parameters
        ----------
        sigma2 : parameter sigma^2 in line 4 of Algorithm 4.1 in MILES paper.
        """
        self.sigma2 = sigma2
        self.return_iip_bag_closest = return_iip_bag_closest
        self.print_progress = print_progress
    
    def fit(self, X, y=None):
        """ Fitting method. Generate Intermediate Instance Pool based on all training bags.

        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]

        Returns
        -------
        self

        """

        # Test if data set has at least three dimensions [bags, instances, features]
        self.check_exceptions(X)

        # Store "intermediate instance pool" (iip). Flat list of all instances of all training bags.
        self.iip = bags2instances(X)

        return self

    def transform(self, X):
        """ Get the bag representation calculating the bag-instance similarity.

        Parameters
        ----------
        X : array-like containing all the bags to do the embedding, 
            with shape [bags, instances, features]

        Returns
        -------
        Bags mapped to instance-based feature space
        
        """

        # Test if data set has at least three dimensions [bags, instances, features]
        self.check_exceptions(X)

        if self.print_progress:
            dist = np.array([self.get_bag_instances_distance(bag) for bag in tqdm(X, total=len(X))])
        else:
            dist = np.array([self.get_bag_instances_distance(bag) for bag in X])

        if self.return_iip_bag_closest:
            return self.similarity_measure(dist[:, 0]), dist[:, 1]
        else:
            return self.similarity_measure(dist[:, 0])

    def get_bag_instances_distance(self, bag):
        """ Calculates the minimum instances between the instances 
            of bag and the training instances

        Parameters
        ----------
        bag : array-like of shape [instances, features]

        Returns
        -------
        the minimum distance between the instance of a bag and the training pool.

        """

        d = [self.get_instance_instances_distance(ins) for ins in bag]
        return np.amin(d, axis=0), np.argmin(d, axis=0)
        
    def get_instance_instances_distance(self, instance):
        """ Calculates the distance between instance, and the instance pool.

        Parameters
        ----------
        instance : array-like of shape [features]

        Returns
        -------
        an array containing the distance between a instance and the instance pool. 
        The distance is the norm between the instance features and the instance pool.

        """
        axes = tuple([e for e in np.arange(1, len(np.array(instance).shape) + 1)])
        return np.linalg.norm(instance - self.iip, axis=axes)
        
    def similarity_measure(self, dist):
        """ Calculates the similarity measure between bags and instances
            used in the MILES paper.

        Parameters
        ----------
        dist : distance measure between bags and instances.

        Returns
        -------
        a matrix containing the distance between each bag and instance in the instance pool.

        """
        return np.exp(-np.array(dist)**2/self.sigma2)
        
    def check_exceptions(self, bags):
        if recursive_len(bags) < 3:
           raise Exception('Dimensionerror: Recursive depth of bags = {}'.format(recursive_len(bags))) 
