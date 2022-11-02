# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['TabData']

# %% ../nbs/00_core.ipynb 2
from fastcore.foundation import patch
from fastcore.utils import *
from fastai.tabular.all import *
from fastai import *
from fastbook import *
import pandas as pd
import numpy as np
import sys
from datetime import date

# %% ../nbs/00_core.ipynb 4
class TabData:
    """ 
        class with main functionality
        has the following instance variables:
            self.data_path
            self.data_var
            self.cont (continuous variables)        
            self.cat (categorical variables)  
            self.tabobj (object of class TabularPandas)
    """
    def __init__(self,
                 data_path: str, # path to find data file; expects csv file
                 dep_var:str=''): # dependent variable
        
        self.data = pd.read_csv(data_path, low_memory=False)
        self.dep_var = dep_var
        self.cont, self.cat = [], []
        self.tabobj = None              
                 

# %% ../nbs/00_core.ipynb 5
@patch
def preprocess_ordinal(self:TabData, 
                       cat_col:str, # categorical columns to order
                       order):
        """
        preprocess ordinal categorical column

        :return: None
        """
        self.data[cat_col] = \
            self.data[cat_col].astype('category')
        self.data[cat_col].cat.set_categories(order,
                            ordered=True, inplace=True)

# %% ../nbs/00_core.ipynb 6
@patch
def log_dep_var(self:TabData):
        """
        Applies log to dependent variable
        
        :return: None
        """
        if self.data is None:
            raise ValueError('data not set')
        if self.dep_var == '':
            raise ValueError('dep_var not set')

        self.data[self.dep_var] = np.log(self.data[self.dep_var])

# %% ../nbs/00_core.ipynb 7
@patch
def exp_dep_var(self:TabData):
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

# %% ../nbs/00_core.ipynb 8
@patch
def convert_to_date(self:TabData, 
                    date_cols:[str]): #list of date columns
        """
        Convert date_cols to date
        
        :return: None
        """
        for col in date_cols:            
            self.data[col] = pd.to_datetime(self.data[col],
                    infer_datetime_format=True).dt.date

# %% ../nbs/00_core.ipynb 9
@patch
def get_train_valid_idxs_time_series(self:TabData, 
                                     year:int, # year 
                                     month:int, # month 
                                     day:int, # day 
                                     date_col): # data column of reference
        """
        
        ATENCAO: VEJA SE O CÓDIGO DESTE MÉTODO ESTÁ CORRETO
        
        Returns training and validation index
        training idx: before date(year, month, day)
        validation idx: from date(year, month, day) onwards
        
        :return (list[int], list[int]): (train_idx, valid_idx)
        """    
        cond = self.data[date_col] < date(2011,11,1)
        train_idx = np.where(cond)[0]
        valid_idx = np.where(~cond)[0]
        
        return (list(train_idx), list(valid_idx))

# %% ../nbs/00_core.ipynb 10
@patch
def handling_dates(self:TabData, 
                   date_cols): # date columns
        """
        Handles dates applying fastai function `add_datepart`
        
        :return: None
        """

        self.data = add_datepart(self.data, date_cols)

# %% ../nbs/00_core.ipynb 11
@patch
def create_tab_obj(self:TabData, 
                   splits=None, # a pair (training indexes, validation indexes) 
                   procs=[Categorify, FillMissing]): # transformations applied to data
        """
        Creates tabobj, instance of TabularPandas
        
        :return: None
        """
        
        self.cont, self.cat = cont_cat_split(self.data, 1, 
                                             dep_var=self.dep_var)
        
        self.tabobj = TabularPandas(self.data, procs, self.cat, self.cont, 
                           y_names=self.dep_var, splits=splits)
