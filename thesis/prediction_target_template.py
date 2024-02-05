'''
Created on 14.04.2021

@author: Tony Culos, Martin Becker
'''
import collections
import numbers
from functools import total_ordering
import warnings

import numpy as np
import scipy.stats
import pandas as pd
import sklearn.metrics

import pysubgroup as ps

@total_ordering
class PredictionTarget(ps.Target, ps.BaseStatisticsCalculator):

    def __init__(
            self,
            target_variable,
            target_estimate):

        self.target_variable = target_variable
        self.target_estimate = target_estimate

    def __repr__(self):
        return "PredictionTarget[" +\
            f"values: {PredictionTarget.get_target_descriptor(self.target_variable)}; " +\
            f"estimates: {PredictionTarget.get_target_descriptor(self.target_estimate)}]"

    @staticmethod
    def get_target_descriptor(target):
        if isinstance(target, np.ndarray):
            return f"ndarray (len={target.shape[0]})"
        elif isinstance(target, (pd.Series, pd.DataFrame)):
            return f"{type(target)} (len={len(target)})"
        # TODO: be more specific about target selection?
        else:
            return f"'{str(target)}'"


class PredictionTargetStatisticsCalculator(ps.CombinedStatisticsCalculator):

    TARGET_STATISTICS = {
        "classification": collections.OrderedDict([
            ("pos", lambda y, _: np.sum(y)),
            ("neg", lambda y, _: len(y) - np.sum(y)),
            ("roc_auc",   sklearn.metrics.roc_auc_score),
            ("avg_prec",  sklearn.metrics.average_precision_score),
            ("precision", lambda y, y_pred: sklearn.metrics.precision_score(y, y_pred > 0.5)),
            ("recall",    lambda y, y_pred: sklearn.metrics.recall_score(y, y_pred > 0.5)),
        ]),
        "regression": collections.OrderedDict([
            ("y_mean",     lambda y, _: np.mean(y)),
            ("y_std",      lambda y, _: np.std(y)),
            ("y_median",   lambda y, _: np.median(y)),
            ("y_mean",     lambda _, y_pred: np.mean(y_pred)),
            ("y_std",      lambda _, y_pred: np.std(y_pred)),
            ("y_median",   lambda _, y_pred: np.median(y_pred)),
            ("pearson_r",  lambda y, y_pred: scipy.stats.pearsonr(y, y_pred)[0]),
            ("pearson_p",  lambda y, y_pred: scipy.stats.pearsonr(y, y_pred)[1]),
            ("spearman_r", lambda y, y_pred: scipy.stats.spearmanr(y, y_pred)[0]),
            ("spearman_p", lambda y, y_pred: scipy.stats.spearmanr(y, y_pred)[1]),
            ("mae",        sklearn.metrics.mean_absolute_error),
            ("mse",        sklearn.metrics.mean_squared_error),
        ]),
    }

    def __init__(
            self,
            target_variable,
            target_estimate,
            target_statistics,
            handle_statistics_errors=np.nan,
            dual_statistics=True) -> None:

        base_statistics_calculator = ps.BaseStatisticsCalculator()

        if isinstance(target_statistics, str):
             target_statistics = PredictionTargetStatisticsCalculator.TARGET_STATISTICS[target_statistics]

        def wrap_statistics_func(func):
            def wrapped(subgroup, df_sg, df):
                return func(
                    PredictionTargetStatisticsCalculator.get_values(target_variable, df_sg), 
                    PredictionTargetStatisticsCalculator.get_values(target_estimate, df_sg))
            return wrapped

        wrapped_statistics = collections.OrderedDict([
            (stat_name, wrap_statistics_func(stat_func))
            for stat_name, stat_func in target_statistics.items()])

        target_statistics_calculator = ps.GenericStatisticsCalculator(
            wrapped_statistics, 
            handle_statistics_errors=handle_statistics_errors, 
            dual_statistics=dual_statistics)

        self.statistics = target_statistics
        self.wrapped_statistics = wrapped_statistics
        super().__init__(base_statistics_calculator, target_statistics_calculator)


    @staticmethod
    def get_values(target, data):
        """TODO: should we drop this and force data frames?"""
        if isinstance(target, np.ndarray):
            return target
        elif isinstance(target, (pd.Series, pd.DataFrame)):
            return target.values()
        # TODO: be more specific about target selection?
        else:
            return data[target].values

    @staticmethod
    def from_target(
            target: PredictionTarget,
            target_statistics,
            handle_statistics_errors=np.nan,
            dual_statistics=True):
        return PredictionTargetStatisticsCalculator(
            target_variable=target.target_variable,
            target_estimate=target.target_estimate,
            target_statistics=target_statistics,
            handle_statistics_errors=handle_statistics_errors,
            dual_statistics=dual_statistics)


class PredictionQFNumeric(ps.BoundedInterestingnessMeasure):

    tpl = collections.namedtuple('PredictionQFNumeric_parameters', ('size', 'metric'))

    def __init__(
            self,
            metric,
            metric_transform=None,
            size_exponent=1,
            relative_subgroup_size=True,
            relative_subgroup_metric='none',
            optimistic_estimate=None,
            handle_metric_error='raise',
            warn_on_metric_error=True,
            prune_lower_than_dataset_metric=False):

        if not isinstance(size_exponent, numbers.Number):
            raise ValueError(f'a is not a number. Received a={size_exponent}')

        self.metric = metric
        self.metric_transform = metric_transform
        self.size_exponent = size_exponent

        self.relative_subgroup_size = relative_subgroup_size
        self.relative_subgroup_metric = relative_subgroup_metric

        if optimistic_estimate is None:
            self.estimator = PredictionQFNumeric.MaxValueOptimisticEstimator(float("inf"))
        elif isinstance(size_exponent, numbers.Number):
            self.estimator = PredictionQFNumeric.MaxValueOptimisticEstimator(optimistic_estimate)
        else:
            self.estimator =  optimistic_estimate

        self.handle_metric_error = handle_metric_error
        self.warn_on_metric_error = warn_on_metric_error

        self.prune_lower_than_dataset_metric = prune_lower_than_dataset_metric

        self.required_stat_attrs = ('size_sg', 'metric_sg')

        # place holders for statistics
        self.dataset_statistics = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target: PredictionTarget):
        """
        Calculate statistics that do not change during the search and can thus be calculated before the actual search starts.
        """

        size_dataset = len(data)

        target_variable_dataset = PredictionTargetStatisticsCalculator.get_values(target.target_variable, data)
        target_estimate_dataset = PredictionTargetStatisticsCalculator.get_values(target.target_estimate, data)

        target_metric_dataset = self.calculate_metric(
            self.metric, target_variable_dataset, target_estimate_dataset)
        self.dataset_statistics = PredictionQFNumeric.tpl(size_dataset, target_metric_dataset)
        
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        
        # TODO: ensure constant statistics
        size_dataset, _ = self.dataset_statistics

        cover_arr, size_sg = ps.get_cover_array_and_size(
            subgroup, data_len=size_dataset, data=data)

        # target values
        target_values_dataset = PredictionTargetStatisticsCalculator.get_values(target.target_variable, data)
        target_values_sg = target_values_dataset[cover_arr]

        # target estimates
        target_estimates_dataset = PredictionTargetStatisticsCalculator.get_values(target.target_estimate, data)
        target_estimates_sg = target_estimates_dataset[cover_arr]

        # calculate subgroup metric
        metric_sg = self.calculate_metric(
            self.metric, target_values_sg, target_estimates_sg)

        return PredictionQFNumeric.tpl(size_sg, metric_sg)

    def evaluate(self, subgroup, target, data, statistics=None):

        statistics_sg = self.ensure_statistics(subgroup, target, data, statistics)
        statistics_dataset = self.dataset_statistics

        q = PredictionQFNumeric.prediction_qf_numeric(
            size_sg=statistics_sg.size,
            metric_sg=statistics_sg.metric,
            size_dataset=statistics_dataset.size,
            metric_dataset=statistics_dataset.metric,
            size_exponent=self.size_exponent,
            metric_transform=self.metric_transform,
            relative_metric=self.relative_subgroup_metric,
            relative_size=self.relative_subgroup_size,
            prune_lower_than_dataset_metric=self.prune_lower_than_dataset_metric)
            
        return q

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        
        statistics_sg = self.ensure_statistics(subgroup, target, data, statistics)
        statistics_dataset = self.dataset_statistics

        # TODO: for some reason in other implementations calling the estimator is moved to `calculate_statistics`; efficiency? would the same apply for `evaluate`?
        return self.estimator.get_estimate(
            size_sg=statistics_sg.size,
            metric_sg=statistics_sg.metric,
            size_dataset=statistics_dataset.size,
            metric_dataset=statistics_dataset.metric,
            size_exponent=self.size_exponent,
            metric_transform=self.metric_transform,
            relative_metric=self.relative_subgroup_metric,
            relative_size=self.relative_subgroup_size)

    def calculate_metric(self, metric, target_values, target_estimate):
        try:
            return metric(target_values, target_estimate)
        except Exception as e:
            if self.handle_metric_error == "raise":
                raise e
            else:
                if self.warn_on_metric_error:
                    warnings.warn(f"Error while evaluating metric: {e}")
                return self.handle_metric_error

    @staticmethod
    def prediction_qf_numeric(
            size_sg,
            metric_sg,
            size_dataset=None,
            metric_dataset=None,
            size_exponent=1,
            metric_transform=None,  # 'reverse', 'invert', or callable
            relative_size=True,
            # TODO: merge with `metric_transform`?
            relative_metric='none',  # 'ratio' or 'difference'
            prune_lower_than_dataset_metric=False
        ):

        if relative_size:
            size = size_sg / size_dataset

        if relative_metric == "none":
            metric = metric_sg
        elif relative_metric == "ratio":
            metric = metric_sg / metric_dataset
        elif relative_metric == "difference":
            metric = metric_sg - metric_dataset
        else:
            raise ValueError(f"Unknown relative metric mode: {relative_metric}")

        if prune_lower_than_dataset_metric:
            if metric_sg < metric_dataset:
                return float("-inf")

        # transfrom metric
        # TODO: reevaluate transform situation!
        if metric_transform is None:
            metric = metric_sg

        # elif isinstance(metric_transform, str) and metric_transform.startswith("invert"):
        #     if metric != 0:
        #         metric = 1.0 / metric_sg
        #     else:
        #         if metric_transform.endswith("invert") or metric_transform.endwith("+"):
        #             metric = float("inf") # TODO: when metric_sg = 0 and inverted just return inf, this assumes low metric is bad
        #         elif metric_transform.endswith("-"):
        #             metric = float("-inf")
        #         else:
        #             raise ValueError(f"Unknown inversion command: {metric_transform}")

        elif callable(metric_transform):
            metric = metric_transform(metric_sg, metric_dataset)

        else:
            raise ValueError(f"Unknown metric transform: {metric_transform}")

        # calculate quality
        return size ** size_exponent * metric
        

    class MaxValueOptimisticEstimator:
        def __init__(self, value):
            self.max_value = value

        def get_data(self, data):
            return data

        def get_estimate(
                self,
                size_sg,
                metric_sg,
                size_dataset=1,
                metric_dataset=1,
                size_exponent=1,
                metric_transform=None,
                relative_size=True, 
                relative_metric=False):
            
            return PredictionQFNumeric.prediction_qf_numeric(
                size_sg=size_sg,
                metric_sg=self.max_value,
                size_dataset=size_dataset,
                metric_dataset=metric_dataset,
                size_exponent=size_exponent,
                metric_transform=metric_transform,
                relative_metric=relative_metric,
                relative_size=relative_size,
                prune_lower_than_dataset_metric=False)


def average_ranking_loss(y_true, y_pred):
    """
    'Average Ranking Loss' as proposed to use in:
    Duivesteijn & Thaele, 2014; Understanding Where Your Classifier Does (Not) Work - The SCaPE Model Class for EMM
    """
    sorted_true = y_true[np.argsort(y_pred)]
    numerator_sum = 0
    for i in range(len(y_true)):
        if sorted_true[i] == 1:
            numerator_sum += (sorted_true[:i+1] == 0).sum()
    return numerator_sum / y_true.sum()
