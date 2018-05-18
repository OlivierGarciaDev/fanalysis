# -*- coding: utf-8 -*-

""" mca module
"""

# Author: Olivier Garcia <o.garcia.dev@gmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from fanalysis.base import Base


class MCA(Base):
    """ Multiple Correspondence Analysis (MCA)
    
    This class inherits from the Base class.
    
    MCA performs a Multiple Correspondence Analysis, given a table of
    categorical variables ; shape= n_rows x n_vars.
    Here n_columns = n_categories = the number of categories that are
    extracted from the data table.

    This implementation only works for dense arrays.

    Parameters
    ----------
    n_components : int, float or None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
          that the amount of variance that needs to be explained is
          greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
              equal to n_components.
            - If n_components is float, select the higher number of
              components lower than n_components.
        
    row_labels : array of strings or None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.
    
    var_labels : array of strings or None
        - If var_labels is an array of strings : this array provides the 
          variable labels.
              If the shape of the array doesn't match with the number of 
              variables : labels are automatically computed for each
              variable.
        - If var_labels is None : labels are automatically computed for
          each variable.
    
    stats : bool
        - If stats is true : stats are computed : contributions and
          square cosines for rows and columns
          (here columns = categories).
        - If stats is false : stats are not computed.

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.
    
    row_labels_ : array of strings
        Labels for the rows.
    
    var_labels : array of strings
        Labels for the variables.

    col_labels_ : array of strings
        Labels for the columns (here columns = categories).
        They are prefixed by the names of the variables.
    
    prefixes_ : array of strings
        Prefixes for the elements of col_labels_

    col_labels_short_ : array of strings
        Short labels for the columns (here columns = categories).
        They are not prefixed by the names of the variables).
    
    eig_ : array of float
        A 3 x n_components_ matrix containing all the eigenvalues
        (1st row), the percentage of variance (2nd row) and the
        cumulative percentage of variance (3rd row).
    
    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.
    
    col_coord_ : array of float
        A n_categories_ x n_components_ matrix containing the column 
        coordinates (= the categories coordinates).
        
    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    col_contrib_ : array of float
        A n_categories_ x n_components_ matrix containing the column 
        contributions (= the categories contributions).
    
    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_cos2_ : array of float
        A n_categories_ x n_components_ matrix containing the columns
        cosines (= the categories cosines).

    n_vars_ : float
        Number of variables in the data table.
        
    n_categories_ : float
        Number of categories that are extracted from the data table.

    n_ : float
        Here n_ = n_rows x n_vars
    
    r_ : float
        Absolute frequencies for the rows = n_vars
    
    c_ : float
        Absolute frequencies for the categories.
    
    model_ : string
        The model fitted = 'mca'
    """
    def __init__(self, n_components=None, row_labels=None, var_labels=None,
                 stats=True):
        Base.__init__(self, n_components, row_labels, None, stats)
        self.var_labels = var_labels
    
    def fit(self, X, y=None):
        """ Fit the model to X.
    
        Parameters
        ----------
        X : array of string, int or float, shape (n_rows, n_vars)
            Training data, where n_rows in the number of rows and n_vars
            is the number of variables.
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.
        
        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Create a dummy variables table
        X_dummies = self._binarization(X)
        
        # Fit a Factorial Analysis to the dummy variables table
        self.r_ = np.sum(X_dummies, axis=1).reshape(-1, 1)
        Base.fit(self, X_dummies, y=None)
        
        # Adjustment of the number of components
        n_eigen = self.n_categories_ - self.n_vars_
        if (self.n_components_ > n_eigen):
            self.n_components_ = n_eigen
            self.eig_ = self.eig_[:, :self.n_components_]
            self.row_coord_ = self.row_coord_[:, :self.n_components_]
            self.col_coord_ = self.col_coord_[:, :self.n_components_]
            if self.stats:
                self.row_contrib_ = self.row_contrib_[:, :self.n_components_]
                self.col_contrib_ = self.col_contrib_[:, :self.n_components_]
                self.row_cos2_ = self.row_cos2_[:, :self.n_components_]
                self.col_cos2_ = self.col_cos2_[:, :self.n_components_]

        # Set col_labels_short_
        self.col_labels_short_ = self.col_labels_short_temp_
        
        # Set col_labels_
        self.col_labels_ = self.col_labels_temp_        
        
        self.model_ = "mca"
        
        return self

    def transform(self, X, y=None):
        """ Apply the dimensionality reduction on X. X is projected on
        the first axes previous extracted from a training set.

        Parameters
        ----------
        X : array of string, int or float, shape (n_rows_sup, n_vars)
            New data, where n_rows_sup is the number of supplementary
            row points and n_vars is the number of variables.
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.
            X rows correspond to supplementary row points that are
            projected onto the axes.
        
        y : None
            y is ignored.

        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points onto the axes.
        """
        # Build dummy variables for the supplementary rows table
        nrows = X.shape[0]
        #ncols = self.col_labels_.shape[0]
        ncols = len(self.col_labels_)
        Y = np.zeros(shape=(nrows, ncols))
        for i in np.arange(0, nrows, 1):
            values = [self.prefixes_[k] + str(X[i, k])
                      for k in np.arange(0, self.n_vars_)]
            for j in np.arange(0, ncols, 1):
                if self.col_labels_[j] in values:
                    Y[i, j] = 1
        
        # Apply the transform method to Y
        return Base.transform(self, Y)

    def _binarization(self, X):
        """ Create a dummy variables table
        
        This function also sets columns prefixes and
        self.col_labels_short_, which is useful for some graphs.
        
        Parameters
        ----------
        X : array of string, int or float, shape (n_rows, n_vars)
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.

        Returns
        -------
        X_d : object
            Returns the dummy variables table.
        
        """
        # Set columns prefixes
        self.n_vars_ = X.shape[1]
        if self.var_labels is None:
            self.prefixes_ = ["col" + str(x) + "_"
                              for x in np.arange(0, self.n_vars_)]
        elif len(self.var_labels) != self.n_vars_:
            self.prefixes_ = ["col" + str(x) + "_"
                              for x in np.arange(0, self.n_vars_)]
        else:
            self.prefixes_ = [str(x) + "_" for x in self.var_labels]
                
        # Dummy variables creation
        X_d = np.empty(shape=(X.shape[0], 0))
        self.col_labels_short_temp_ = np.empty(shape=(0,))
        self.col_labels_temp_ = np.empty(shape=(0,))
        for i in range(X.shape[1]):
            lb = LabelBinarizer()
            lb.fit(X[:, i])
            X_di = lb.transform(X[:, i])
            if lb.classes_.shape[0] == 2:
                if X_di[0,0] == (X[0, i] == lb.classes_[0]):
                    X_di = np.c_[X_di, 1 - X_di]
                else:
                    X_di = np.c_[1 - X_di, X_di]
            X_d = np.append(X_d, X_di, axis=1)
            self.col_labels_short_temp_ = np.append(
                                            self.col_labels_short_temp_,
                                            lb.classes_)
            self.col_labels_temp_ = np.append(self.col_labels_temp_,
                                              [self.prefixes_[i]
                                              + str(x)
                                              for x in lb.classes_]
                                              )

        self.n_categories_ = X_d.shape[1]
        
        # Return the dummy variables table
        return X_d
