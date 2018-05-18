# -*- coding: utf-8 -*-

""" pca module
"""

# Author: Olivier Garcia <o.garcia.dev@gmail.com>
# License: BSD 3 clause

import numpy as np
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fanalysis.base import Base


class PCA(Base):
    """ Principal Components Analysis (PCA)
    
    This class inherits from the Base class.
    
    PCA performs a Principal Components Analysis, given a table of
    numeric variables ; shape= n_rows x n_columns.

    This implementation only works for dense arrays.

    Parameters
    ----------
    std_unit : bool
       - If True : the data are scaled to unit variance.
       - If False : the data are not scaled to unit variance.
    
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
    
    eigen_vectors_ : array of float
        Eigen vectors extracted from the Principal Components Analysis.
    
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
    
    col_cor_ : array of float
        A n_columns x n_components_ matrix containing the correlations
        between variables (= columns) and axes.
    
    means_ : array of float
        The mean for each variable (= for each column).

    std_ : array of float
        The standard deviation for each variable (= for each column).
    
    ss_col_coord_ : array of float
        The sum of squared of columns coordinates.

    model_ : string
        The model fitted = 'pca'
    """
    def __init__(self, std_unit=True, n_components=None, row_labels=None,
                 col_labels=None, stats=True):
        self.std_unit = std_unit
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.stats = stats

    def fit(self, X, y=None):
        """ Fit the model to X

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns
            (= the number of variables).
            X is a table containing numeric values.
        
        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Stats initialization
        self.row_contrib_ = None
        self.col_contrib_ = None
        self.row_cos2_ = None
        self.col_cos2_ = None
        
        # Compute SVD
        self._compute_svd(X)
        
        return self

    def transform(self, X, y=None):
        """ Apply the dimensionality reduction on X.

        X is projected on the first axes previous extracted from a
        training set.

        Parameters
        ----------
        X : array of float, shape (n_rows_sup, n_columns)
            New data, where n_rows_sup is the number of supplementary
            row points and n_columns is the number of columns.
            X rows correspond to supplementary row points that are
            projected on the axes.
            X is a table containing numeric values.
        
        y : None
            y is ignored.

        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points on the axes.
        """
        if self.std_unit:
            Z = (X - self.means_) / self.std_
        else:
            Z = X - self.means_
        
        return Z.dot(self.eigen_vectors_)
        
    def _compute_svd(self, X):
        """ Compute a Singular Value Decomposition
        
        Then, this function computes :
            n_components_ : number of components.
            eig_ : eigen values.
            eigen_vectors_ : eigen vectors.
            row_coord_ : row coordinates.
            col_coord_ : column coordinates.
            _compute_stats(X) : if stats_ is True.
            row_labels_ : row labels.
            col_labels_ : columns labels.

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows is the number of rows and
            n_columns is the number of columns.
            X is a table containing numeric values.

        Returns
        -------
        None
        """
        # Initializations
        self.means_ = np.mean(X, axis=0).reshape(1, -1)
        if self.std_unit:
            self.std_ = np.std(X, axis=0, ddof=0).reshape(1, -1)
            Z = (X - self.means_) / self.std_
        else:
            Z = X - self.means_        
                
        # SVD
        U, lambdas, V = np.linalg.svd(Z, full_matrices=False)
        
        # Eigen values - first step
        eigen_values = lambdas ** 2 / Z.shape[0]
        eigen_values_percent = 100 * eigen_values / np.sum(eigen_values)
        eigen_values_percent_cumsum = np.cumsum(eigen_values_percent)
        
        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = len(eigen_values)
        elif (self.n_components_ >= 0) and (self.n_components_ < 1):
            i = 0
            threshold = 100 * self.n_components_
            while eigen_values_percent_cumsum[i] < threshold:
                i = i + 1
            self.n_components_ = i
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values))
                and (isinstance(self.n_components_, int))):
            self.n_components_ = int(np.trunc(self.n_components_))
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values))
                and (isinstance(self.n_components_, float))):
            self.n_components_ = int(np.floor(self.n_components_))
        else:
            self.n_components_ = len(eigen_values)
        
        # Eigen values - second step
        self.eig_ = np.array([eigen_values[:self.n_components_],
                             eigen_values_percent[:self.n_components_],
                             eigen_values_percent_cumsum[:self.n_components_]])
        
        # Eigen vectors
        self.eigen_vectors_ = V.T[:, :self.n_components_]
        
        # Factor coordinates for rows - first step
        row_coord = U * lambdas.reshape(1, -1)

        # Factor coordinates for columns - first step
        col_coord = V.T.dot(np.diag(eigen_values**(0.5)))
        self.ss_col_coord_ = (np.sum(col_coord ** 2, axis=1)).reshape(-1, 1)

        # Factor coordinates for rows - second step
        self.row_coord_ = row_coord[:, :self.n_components_]

        # Factor coordinates for columns - second step
        self.col_coord_ = col_coord[:, :self.n_components_]

        # Compute stats
        if self.stats:
            self._compute_stats(X, Z)
        
        # Set row labels
        nrows = X.shape[0]
        self.row_labels_ = self.row_labels
        if (self.row_labels_ is None) or (len(self.row_labels_) != nrows):
            self.row_labels_ = ["row" + str(x) for x in np.arange(0, nrows)]
        
        # Set column labels
        ncols = X.shape[1]
        self.col_labels_ = self.col_labels
        if (self.col_labels_ is None) or (len(self.col_labels_) != ncols):
            self.col_labels_ = ["col" + str(x) for x in np.arange(0, ncols)]
        self.col_labels_short_ = self.col_labels_
        self.model_ = "pca"

    def _compute_stats(self, X, Z):
        """ Compute statistics : 
                row_contrib_ : row contributions.
                col_contrib_ : column contributions.
                row_cos2_ : row cosines.
                col_cos2_ : column cosines.
                col_cor_ : correlations between variables (= columns)
                and axes.
        
        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows is the number of rows and
            n_columns is the number of columns.
            X is a table containing numeric values.
        Z : array of float, shape (n_rows, n_columns)
            Transformed training data (centered and -optionnaly- scaled
            values)
            where n_rows is the number of rows and n_columns is the
            number of columns.

        Returns
        -------
        None
        """
        # Contributions
        n = Z.shape[0]
        row_contrib = 100 * ((1 / n)
                          * (self.row_coord_ ** 2)
                          * (1 / self.eig_[0].T))
        col_contrib = 100 * (self.col_coord_ ** 2) * (1 / self.eig_[0].T)
        self.row_contrib_ = row_contrib[:, :self.n_components_]
        self.col_contrib_ = col_contrib[:, :self.n_components_]
        
        # Cos2
        row_cos2 = ((self.row_coord_ ** 2)
                    / (np.linalg.norm(Z, axis=1).reshape(-1, 1) ** 2))
        self.row_cos2_ = row_cos2[:, :self.n_components_]
        col_cos2 = (self.col_coord_ ** 2) / self.ss_col_coord_
        self.col_cos2_ = col_cos2[:, :self.n_components_]
        self.ss_col_coord_ = None
        
        # Correlations between variables and axes
        nvars = self.means_.shape[1]
        self.col_cor_ = np.zeros(shape=(nvars, self.n_components_))
        for i in np.arange(0, nvars):
            for j in np.arange(0, self.n_components_):
                self.col_cor_[i, j] = stat.pearsonr(X[:,i],
                                                    self.row_coord_[:,j])[0]
        
    def correlation_circle(self, num_x_axis, num_y_axis, figsize=None):
        """ Plot the correlation circle
        
        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.
        
        num_y_axis : int
             Select the component to plot as y axis.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect="equal")
        ax.add_artist(patches.Circle((0, 0), 1.0, color="black", fill=False))
        
        x_serie = self.col_cor_[:, num_x_axis - 1]
        y_serie = self.col_cor_[:, num_y_axis - 1]
        labels = self.col_labels_
        
        for i in np.arange(0, x_serie.shape[0]):
            x = x_serie[i]
            y = y_serie[i]
            label = labels[i]
            delta = 0.1 if y >= 0 else -0.1
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops={"facecolor": "black",
                                    "width": 1,
                                    "headwidth": 4})
            ax.text(x, y + delta, label,
                    horizontalalignment="center", verticalalignment="center",
                    color="blue")
        
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        plt.title("Correlation circle")
        plt.xlabel("Dim " + str(num_x_axis) + " ("
                    + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")
        plt.ylabel("Dim " + str(num_y_axis) + " ("
                    + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.show()
