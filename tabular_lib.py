import pandas as pd
import numpy as np
from fastcore.utils import *
from fastai.tabular.all import *
from fastai import *
from fastbook import *
import sys
from datetime import date

from sklearn.tree import *
from sklearn.ensemble import *

from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from dtreeviz import *
from dtreeviz.trees import *

class TabData:
    def __init__(self, data_path, dep_var=''):
        """
        :param data_path: path to data (pathlib.Path)
        :param dep_var: dependent variable (str)
        :param cont: list of continuous variables ([str])
        :param cat: list of categorical variables ([str])
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        
        self.dep_var = dep_var
        self.cont, self.cat = [], []
        # object of class TabularPandas
        self.tabobj = None        

    def preprocess_ordinal(self, cat_col, order):
        """
        preprocess ordinal categorical column

        :param cat_col: categorical column to order (str)
        :param order: list of str indicating the order ([str])
        :return: None
        """
        self.data[cat_col] = \
            self.data[cat_col].astype('category')
        self.data[cat_col].cat.set_categories(order,
                            ordered=True, inplace=True)

    def log_dep_var(self):
        """
        Applies log to dependent variable
        :return:
        """
        if self.data is None:
            raise ValueError('data not set')
        if self.dep_var == '':
            raise ValueError('dep_var not set')

        self.data[self.dep_var] = np.log(self.data[self.dep_var])

    def exp_dep_var(self):
        """
        Applies exp to dependent variable
        Inverse of self.exp_dep_var
        :return: None
        """
        if self.data is None:
            raise ValueError('data not set')
        if self.dep_var == '':
            raise ValueError('dep_var not set')

        self.data[self.dep_var] = np.log(self.data[self.dep_var])
        
    def convert_to_date(self, date_cols):
        """
        Convert date_cols to date
        :param date_cols: list of columns to convert to date, 
            type [str]
        :return: None
        """
        for col in date_cols:            
            self.data[col] = pd.to_datetime(self.data[col],
                    infer_datetime_format=True).dt.date
    
    def get_train_valid_idxs_time_series(self, year, month, day, date_col):
        """
        Returns training and validation index
        training idx: before date(year, month, day)
        validation idx: from date(year, month, day) onwards
        :param year int: year
        :param month int: month
        :param day int: day
        :param date_col str: date column of reference
        :return (list[int], list[int]): (train_idx, valid_idx)
        """    
        cond = self.data[date_col] < date(2011,11,1)
        train_idx = np.where(cond)[0]
        valid_idx = np.where(~cond)[0]
        
        return (list(train_idx), list(valid_idx))    

    def handling_dates(self, date_cols):
        """
        Handles dates applying fastai function add_datepart
        :param date_cols:
        :return: None
        """

        self.data = add_datepart(self.data, date_cols)
        
    def create_tab_obj(self, splits=None, procs=[Categorify, FillMissing]):
        """
        Creates tabobj, instance of TabularPandas
        :param splits: pair of lists of indexes; 
            the 1st refers to training; the 2nd refers to validation
        :return: None
        """
        
        self.cont, self.cat = cont_cat_split(self.data, 1, 
                                             dep_var=self.dep_var)
        
        self.tabobj = TabularPandas(self.data, procs, self.cat, self.cont, 
                           y_names=self.dep_var, splits=splits)
    
    
def tree_viz(model, xs, y, dep_var, sample_sz=500, fontname='DejaVu Sans', 
             scale=1.6,  label_fontsize=10, orientation='UD'):
    """
    vizualization using dtreeviz package
    :param model: can be a DecisionTreeRegressor or
        a DecisionTreeClassifier
    :param sample_sz int: size of sample to vizualize
    """   
    samp_idx = np.random.permutation(len(y))[:sample_sz]

    model.fit(xs, y)        
    viz = dtreeviz(model, xs.iloc[samp_idx], y.iloc[samp_idx], 
             xs.columns, dep_var, fontname=fontname, 
             scale=scale, label_fontsize=label_fontsize, 
             orientation=orientation)    
    return viz
    
class ForestModel:
    """
    This class cannot be instantiated
    The model can be used for modeling purposes
    or for interpretation, or both
    """    
    def __init__(self, tabobj, metric_func):
        """
        :param tabobj: object of class TabularPandas
        :param metric_func: func that takes 2 args and
            returns the metric; the 1st arg is the prediction;
            the 2nd is the correct value
        """
        to = tabobj
        self.xs, self.y = to.train.xs, to.train.y
        self.valid_xs = to.valid.xs
        self.valid_y = to.valid.y
        
        self.model = None    
        self.metric_func = metric_func
     
    def get_metrics(self): 
        """
        This method should return a DataFrame with metrics 
            from training set and validation set
        """        
        pass
    
    def plot_metricXn_estimators(self, label='', xlabel='number of estimators', 
                                 ylabel=''):
        """
        Plots metric Vs n_estimators
        """
        n_est = self.model.n_estimators
        
        preds = np.stack([t.predict(self.valid_xs) \
                          for t in self.model.estimators_])
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot([self.metric_func(preds[:i+1].mean(0), self.valid_y) \
                  for i in range(n_est)])
        
        plt.show()
    
    def feat_importance(self):
        """
        Returns a pd.DataFrame with 2 columns:
            cols: self.xs.column
            imp: self.model.feature_importances_
        """
        return pd.DataFrame({'cols':self.xs.columns, 
                             'imp':self.model.feature_importances_}
                           ).sort_values('imp', ascending=False)
    
    def plot_fi(self, n_features=None, figsize=(12, 7)):
        """
        Plots feature importances
        """
        
        if n_features is None:
            n_features = len(self.xs.columns)           
        
        fi = self.feat_importance()[:n_features]
        
        fi.plot('cols', 'imp', 'barh', 
                figsize=figsize, legend=False)
                
    def keep_cols(self, to_keep):        
        """
        Keep the cols informed in to_keep
        :param to_keep [str]: cols to keep
        """
         
        self.xs = self.xs[to_keep]
        self.valid_xs = self.valid_xs[to_keep] 
        
    def get_oob_error(self):
        """
        Returns the out of bag prediction
        The oob prediction refers only to the training set
        If the oob error is much lower than the validations error
            it means that something else is cusing the error, in 
            addition to normal generalization error
        :return: oob error
        """
        
        return self.metric_func(self.model.oob_prediction_, self.y)
    
    def get_prediction_std(self):
        """
        Returns standard deviation of the predictions given by
        all the predictors
        :returns: np.array(float)
        """
        
        preds = np.stack([t.predict(self.valid_xs) \
                          for t in self.model.estimators_])
        
        return preds.std(0)
    
    def plot_cluster_columns(self):
        """
        Plots cluster columns, which help remove redundant features
        """
        
        cluster_columns(self.xs)
        
    def plot_partial_dependence(self, cols, grid_resolution=20,
                                figsize(12,4)):
        """
        Plots partial dependence plot
        :param cols: list of columns
        """
        
        fig,ax = plt.subplots(figsize=(12, 4))
                
class Regression(ForestModel):
    """
    Subclass of ForestModel
    Used for regression models
    """
    def __init__(self, tabobj, 
                 metric_func=lambda pred, y: \
                     round(math.sqrt(((pred-y)**2).mean()), 6)):
        """
        :param tabobj: object of class TabularPandas
        :param metric_func: func that takes 2 args and
            returns the metric; the 1st arg is the prediction;
            the 2nd is the correct value
        """
        
        super().__init__(tabobj, metric_func)
               
    
    def set_model(self, n_estimators=40, max_samples=200_000,
                  max_features=0.5, min_samples_leaf=5, **kwargs):
        self.model = RandomForestRegressor(n_jobs=-1, 
                         n_estimators=n_estimators, max_samples=max_samples, 
                         max_features=max_features, 
                         min_samples_leaf=min_samples_leaf, 
                         oob_score=True, *kwargs).fit(self.xs, self.y)   
    
    def m_rmse(self, xs, y): 
        pred = self.model.predict(xs)
        
        return self.metric_func(pred, y)
    
    def get_metrics(self): 
        """
        This method should return a dictionary with metrics 
            from training set and validation set
        """
        train = self.m_rmse(self.xs, self.y)
        valid = self.m_rmse(self.valid_xs, self.valid_y)
        
        return pd.DataFrame({'Set':['train', 'validation'], 
                      'RMSE':[train, valid]})
        
def get_oob(xs, y, forest=RandomForestRegressor, n_estimators=40, 
            min_samples_leaf=15, max_samples=50000, max_features=0.5, 
            n_jobs=-1):
        """
        The OOB score is a number returned by sklearn that
        ranges between 1.0 for a perfect model and 0.0 for 
        a random model (R^2).
        """        
        
        m = forest(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, 
                   max_samples=max_samples, max_features=max_features, 
                   n_jobs=n_jobs, oob_score=True)
        m.fit(xs, y)
        
        return m.oob_score_
    
def get_oob_simul(xs, y, cols_to_drop, forest=RandomForestRegressor, 
                  n_estimators=40, min_samples_leaf=15,
                  max_samples=50000, max_features=0.5, n_jobs=-1):
    """
    Returns the OOB score after droping each one of the columns
        in cols_to_drop
    Returns a dict, where the key is the column deleted and the 
        corresponding value is the OOB score
    :return: { deleted_colum:oob_score }
    """

    return {c:get_oob(xs.drop(c, axis=1), y, forest, n_estimators, min_samples_leaf,
                      max_samples, max_features, n_jobs) \
            for c in cols_to_drop }        
        
def get_module(name):
    """
    :param name func: name to find the module
    :return str: name's module
    """
    return sys.modules[name.__module__]    
