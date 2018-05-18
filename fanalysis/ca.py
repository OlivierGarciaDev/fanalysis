# -*- coding: utf-8 -*-

""" ca module
"""

# Author: Olivier Garcia <o.garcia.dev@gmail.com>
# License: BSD 3 clause

import pandas as pd

from fanalysis.base import Base


class CA(Base):
    """ Correspondence Analysis (CA)
    
    This class inherits from the Base class.
    
    CA performs a Correspondence Analysis, given a contingency table
    containing absolute frequencies ; shape= n_rows x n_columns.

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
    
    col_labels : array of strings or None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of 
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.
    
    stats : bool
        - If stats is true : stats are computed : contributions and
          square cosines for rows and columns.
        - If stats is false : stats are not computed.

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.
    
    row_labels_ : array of strings
        Labels for the rows.
    
    col_labels_ : array of strings
        Labels for the columns.
    
    col_labels_short_ : array of strings
        Short labels for the columns.
        Useful only for MCA, which inherits from Base class. In that
        case, the short labels for the columns at not prefixed by the
        names of the variables.
    
    eig_ : array of float
        A 3 x n_components_ matrix containing all the eigenvalues
        (1st row), the percentage of variance (2nd row) and the
        cumulative percentage of variance (3rd row).
    
    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.
    
    col_coord_ : array of float
        A n_columns x n_components_ matrix containing the column
        coordinates.
        
    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.
    
    col_contrib_ : array of float
        A n_columns x n_components_ matrix containing the column
        contributions.
    
    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.
    
    col_cos2_ : array of float
        A n_columns x n_components_ matrix containing the column
        cosines.

    n_ : float
        The sum of the absolute frequencies in the X array.
    
    r_ : float
        The sum of the absolute frequencies in the X array, for each row 
        (along axis = 1).
    c_ : float
        The sum of the absolute frequencies in the X array, for each
        column (along axis = 0).
    
    model_ : string
        The model fitted = 'ca'
    """
    
    def fit(self, X, y=None):
        """ Fit the model to X

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.
        
        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        Base.fit(self, X, y=None)
        self.model_ = "ca"
        return self
