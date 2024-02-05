'''
Created on 29.09.2017

@author: lemmerfn
'''
from collections import namedtuple
from functools import total_ordering
import numpy as np
from sklearn.metrics import roc_auc_score
import pysubgroup as ps
from pysubgroup.subgroup_description import EqualitySelector
from pysubgroup import AbstractInterestingnessMeasure


@total_ordering
class BinaryTarget:

    statistic_types = ('size_sg', 'size_dataset', 'positives_sg', 'positives_dataset', 'size_complement',
                      'relative_size_sg', 'relative_size_complement', 'coverage_sg', 'coverage_complement',
                      'target_share_sg', 'target_share_complement', 'target_share_dataset', 'lift','balance_ratio')

    def __init__(self, target_attribute=None, target_value=None, target_selector=None):
        """
        Creates a new target for the boolean model class (classic subgroup discovery).
        If target_attribute and target_value are given, the target_selector is computed using attribute and value
        """
        if target_attribute is not None and target_value is not None:
            if target_selector is not None:
                raise ValueError("BinaryTarget is to be constructed EITHER by a selector OR by attribute/value pair")
            target_selector = EqualitySelector(target_attribute, target_value)
        if target_selector is None:
            raise ValueError("No target selector given")
        self.target_selector = target_selector

    def __repr__(self):
        return "T: " + str(self.target_selector)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def covers(self, instance):
        return self.target_selector.covers(instance)

    def get_attributes(self):
        return (self.target_selector.attribute_name,)

    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)
        positives = self.covers(data)
        instances_subgroup = size_sg
        positives_dataset = np.sum(positives)
        instances_dataset = len(data)
        positives_subgroup = np.sum(positives[cover_arr])
        negatives_subgroup = instances_subgroup - positives_subgroup
        return instances_dataset, positives_dataset, instances_subgroup, positives_subgroup
    
    def get_AUC(self,a,data,subgroup):
        # self.a =a
        
        auc= AUCQF(a)
        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        y = np.array(self.covers(data)[cover_arr])
        y_prob = np.array(data['y_prob'][cover_arr])
        b = (len(y) ** a) / (len(y)+1)
        auc_score = auc.calculate_auc_score(y, y_prob, b)
        
        return auc_score
        
        

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        #if self.all_statistics_present(cached_statistics):
         #   return cached_statistics

        (instances_dataset, positives_dataset, instances_subgroup, positives_subgroup) = \
            self.get_base_statistics(subgroup, data)
        statistics = {}
        statistics['size_sg'] = instances_subgroup
        statistics['size_dataset'] = instances_dataset
        statistics['positives_sg'] = positives_subgroup
        statistics['positives_dataset'] = positives_dataset
        statistics['size_complement'] = instances_dataset - instances_subgroup
        statistics['relative_size_sg'] = instances_subgroup / instances_dataset
        statistics['relative_size_complement'] = (instances_dataset - instances_subgroup) / instances_dataset
        statistics['coverage_sg'] = positives_subgroup / positives_dataset
        statistics['coverage_complement'] = (positives_dataset - positives_subgroup) / positives_dataset
        statistics['target_share_sg'] = positives_subgroup / instances_subgroup
        if instances_dataset == instances_subgroup:
            statistics['target_share_complement'] = float("nan")
        else:
            statistics['target_share_complement'] = (positives_dataset - positives_subgroup) / (instances_dataset - instances_subgroup)
        statistics['target_share_dataset'] = positives_dataset / instances_dataset
        statistics['lift'] = statistics['target_share_sg'] / statistics['target_share_dataset']
        negatives_subgroup = instances_subgroup - positives_subgroup
        statistics['balance_ratio'] = (positives_subgroup / negatives_subgroup)
        statistics['auc_score'] = auc_score
        return statistics


class SimplePositivesQF(ps.AbstractInterestingnessMeasure):  # pylint: disable=abstract-method
    tpl = namedtuple('PositivesQF_parameters', ('size_sg', 'positives_count'))

    def __init__(self):
        self.dataset_statistics = None
        self.positives = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ('size_sg', 'positives_count')

    def calculate_constant_statistics(self, data, target):
        assert isinstance(target, BinaryTarget)
        self.positives = target.covers(data)
        self.dataset_statistics = SimplePositivesQF.tpl(len(data), np.sum(self.positives))
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, statistics=None): # pylint: disable=unused-argument
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(self.positives), data)
        return SimplePositivesQF.tpl(size_sg, np.count_nonzero(self.positives[cover_arr]))

# <<< GpGrowth >>>
    def gp_get_stats(self, row_index):
        return np.array([1, self.positives[row_index]], dtype=int)

    def gp_get_null_vector(self):
        return np.zeros(2)

    def gp_merge(self, l, r):
        l += r

    def gp_get_params(self, _cover_arr, v):
        return SimplePositivesQF.tpl(v[0], v[1])

    def gp_to_str(self, stats):
        return " ".join(map(str, stats))

    def gp_size_sg(self, stats):
        return stats[0]

    @property
    def gp_requires_cover_arr(self):
        return False
    

# TODO Make ChiSquared useful for real nominal data not just binary
#      Introduce Enum for direction
#      Maybe it is possible to give a optimistic estimate for ChiSquared

class StandardQF(SimplePositivesQF, ps.BoundedInterestingnessMeasure):
    """
    StandardQF which weights the relative size against the difference in averages

    The StandardQF is a general form of quality function which for different values of a is order equivalen to
    many popular quality measures.

    Attributes
    ----------
    a : float
        used as an exponent to scale the relative size to the difference in averages

    """

    @staticmethod
    def standard_qf(a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        if not hasattr(instances_subgroup, '__array_interface__') and (instances_subgroup == 0):
            return np.nan
        p_subgroup = np.divide(positives_subgroup, instances_subgroup)
        #if instances_subgroup == 0:
        #    return 0
        #p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        return (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)

    def __init__(self, a):
        """
        Parameters
        ----------
        a : float
            exponent to trade-off the relative size with the difference in means
        """
        self.a = a
        super().__init__()

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQF.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg, statistics.positives_count)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQF.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.positives_count, statistics.positives_count)

    def optimistic_generalisation(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        pos_remaining = dataset.positives_count - statistics.positives_count
        return StandardQF.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg + pos_remaining, dataset.positives_count)
    

from pysubgroup import AbstractInterestingnessMeasure

class ARLQF(AbstractInterestingnessMeasure):
    
    
    def calculate_arl_score(self, y, y_prob,b):
  
        sorted_indices = np.argsort(y)
        sorted_true = y[sorted_indices]
        sorted_ranking = y_prob[sorted_indices]
    
        true_indices = np.where(sorted_true == 1)[0]
        false_indices = np.where(sorted_true == 0)[0]
    
        false_rankings = sorted_ranking[false_indices]
        true_rankings = sorted_ranking[true_indices]
    
     
        false_rankings_broadcasted = np.expand_dims(false_rankings, axis=1)
        true_rankings_broadcasted = np.expand_dims(true_rankings, axis=0)
    
        higher_rankings = false_rankings_broadcasted > true_rankings_broadcasted
        equal_rankings = false_rankings_broadcasted == true_rankings_broadcasted
    
        PENNi_sum = np.sum(higher_rankings, axis=1) + 0.5 * np.sum(equal_rankings, axis=1)
    
        
        sorted_true_true_indices = sorted_true[true_indices][:, np.newaxis]
    
       
        numerator_sum = np.sum(sorted_true_true_indices * PENNi_sum)
        denominator_sum = np.sum(y)

    
        if denominator_sum == 0 or np.isnan(denominator_sum):
            arl_score = np.nan
        else:
            arl_score = (numerator_sum / denominator_sum) *b
        return arl_score

    def __init__(self,a):
        
        self.required_stat_attrs = ('size_sg', 'positives_count')
        self.a = a

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        y = np.array(target.covers(data)[cover_arr])
        y_prob = np.array(data['y_prob'][cover_arr])
        a = self.a
        b = (len(y) ** a) /(((len(data['y_prob']) **2)+1))
        arl_score = self.calculate_arl_score(y, y_prob,b)
        return arl_score

    def calculate_constant_statistics(self, data, target):
        assert isinstance(target, BinaryTarget)
        self.positives = target.covers(data)
        self.dataset_statistics = SimplePositivesQF.tpl(len(data), np.sum(self.positives))
        self.has_constant_statistics = True
    
    def evaluate(self, subgroup, target, data, statistics=None):
        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        y = np.array(target.covers(data)[cover_arr])
        y_prob = np.array(data['y_prob'][cover_arr])
        a = self.a
        b = (len(y) ** a) /(((len(data['y_prob']) **2)+1))
        arl_score = self.calculate_arl_score(y, y_prob,b)
        return arl_score
    
class AUCQF(AbstractInterestingnessMeasure):

    def calculate_auc_score(self, y, y_prob, b):
        unique_classes = np.unique(y)
        
        if len(unique_classes) < 2:
            auc_score = np.nan
        else:
            auc_score = roc_auc_score(y, y_prob) * b
        
        return auc_score

    def __init__(self, a):
        self.required_stat_attrs = ('size_sg', 'positives_count')
        self.a = a

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        y = np.array(target.covers(data)[cover_arr])
        y_prob = np.array(data['y_prob'][cover_arr])
        a = self.a
        b = a
        auc_score = self.calculate_auc_score(y, y_prob, b)
        return auc_score

    def calculate_constant_statistics(self, data, target):
        assert isinstance(target, BinaryTarget)
        self.positives = target.covers(data)
        self.dataset_statistics = SimplePositivesQF.tpl(len(data), np.sum(self.positives))
        self.has_constant_statistics = True
    
    def evaluate(self, subgroup, target, data, statistics=None):
        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        y = np.array(target.covers(data)[cover_arr])
        y_prob = np.array(data['y_prob'][cover_arr])
        a = self.a
        b = (len(y) ** a) / (len(y)+1)
        auc_score = self.calculate_auc_score(y, y_prob, b)
        return auc_score
