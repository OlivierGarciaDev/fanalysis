# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd

from fanalysis.mca import MCA


class TestMca(unittest.TestCase):
    """ Unit tests for the MCA class
    """
    def test_mca(self):
        """ Test for the fit_transfom operation - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._fit_transform_comparison(data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._fit_transform_comparison(data_type = dt,
                                               n_components = i)

    def test_eigen_(self):
        """ Test for the eigen values - Comparison with the R FactoMiner
        output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("eig_", "mca_eig.txt", data_type = dt,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("eig_", "mca_eig.txt", data_type = dt,
                                     n_components = i)
    
    def test_row_coord_(self):
        """ Test for the rows coordinates - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("row_coord_", "mca_row_coord.txt",
                                 data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("row_coord_", "mca_row_coord.txt",
                                     data_type = dt, n_components = i)

    def test_row_contrib_(self):
        """ Test for the rows contributions - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("row_contrib_", "mca_row_contrib.txt",
                                 data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("row_contrib_", "mca_row_contrib.txt",
                                     data_type = dt, n_components = i)
    
    def test_row_cos2_(self):
        """ Test for the rows cos2 - Comparison with the R FactoMiner
        output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("row_cos2_", "mca_row_cos2.txt",
                                 data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("row_cos2_", "mca_row_cos2.txt",
                                     data_type = dt, n_components = i)
    
    def test_col_coord_(self):
        """ Test for the columns coordinates - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("col_coord_", "mca_col_coord.txt",
                                 data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_coord_", "mca_col_coord.txt",
                                     data_type = dt, n_components = i)

    def test_col_contrib_(self):
        """ Test for the columns contributions - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("col_contrib_", "mca_col_contrib.txt",
                                 data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_contrib_", "mca_col_contrib.txt",
                                     data_type = dt, n_components = i)
    
    def test_col_cos2_(self):
        """ Test for the columns cos2 - Comparison with the R FactoMiner
        output
        """
        for dt in ["string", "numeric"]:
            self._X_Y_comparison("col_cos2_", "mca_col_cos2.txt",
                                 data_type = dt, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_cos2_", "mca_col_cos2.txt",
                                     data_type = dt, n_components = i)

    def test_row_topandas(self):
        """ Test for the row_topandas method - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._row_topandas_comparison(data_type = dt, n_components = None,
                                          row_labels = False)
            self._row_topandas_comparison(data_type = dt, n_components = None,
                                          row_labels = True)
            for i in np.arange(-10, 10, 0.5):
                self._row_topandas_comparison(data_type = dt, n_components = i,
                                              row_labels = False)
                self._row_topandas_comparison(data_type = dt, n_components = i,
                                              row_labels = True)
    
    def test_col_topandas(self):
        """ Test for the col_topandas method - Comparison with the
        R FactoMiner output
        """
        for dt in ["string", "numeric"]:
            self._col_topandas_comparison(data_type = dt, n_components = None,
                                          var_labels = False)
            self._col_topandas_comparison(data_type = dt, n_components = None,
                                          var_labels = True)
            for i in np.arange(-10, 10, 0.5):
                self._col_topandas_comparison(data_type = dt, n_components = i,
                                              var_labels = False)
                self._col_topandas_comparison(data_type = dt, n_components = i,
                                              var_labels = True)
    
    def _fit(self, data_type, n_components):
        """ This function fits the model to the data """
        if data_type == "string":
            df = pd.read_table("fanalysis/tests/mca_data_string.txt",
                               header=0, index_col=0, delimiter="\t")
        else :
            df = pd.read_table("fanalysis/tests/mca_data_numeric.txt",
                               header=0, index_col=0, delimiter="\t")
        M = df.as_matrix()
        mca = MCA(n_components = n_components)
        mca.fit(M)
        return mca

    def _adjust_n_components(self, n_components, eigen_values):
        """ This function sets relevant values for n_components """
        if (n_components is None):
            n_components = eigen_values.shape[1]
        elif (n_components >= 0) and (n_components < 1):
            i = 0
            threshold = 100 * n_components
            while eigen_values[2, i] < threshold:
                i = i + 1
            n_components = i
        elif ((n_components >= 1)
                and (n_components <= eigen_values.shape[1])
                and (isinstance(n_components, int))):
            n_components = int(np.trunc(n_components))
        elif ((n_components >= 1)
                and (n_components <= eigen_values.shape[1])
                and (isinstance(n_components, float))):
            n_components = int(np.floor(n_components))
        else:
            n_components = eigen_values.shape[1]
        return n_components
    
    def _compute_Y(self, X, Y_temp, attr):
        """ This function sets the signs of the coordinates to those of
        the R FactoMineR output
        """
        if (attr == "row_coord_") or (attr == "col_coord_"):
            x = X[1, :]
            y = Y_temp[1, :]
            z = x * y
            for i in np.arange(0, z.shape[0]):
                z[i] = 1 if z[i] >=0 else -1
            return Y_temp * z.reshape(1, -1)
            
        else:
            return Y_temp

    def _fit_transform_comparison(self, data_type, n_components=None):
        """ This function compares the result of the fit_transform
        operation with the R FactoMineR output
        """
        if n_components is None:
            mca1 = MCA()
            mca2 = MCA()
        else:
            mca1 = MCA(n_components = n_components)
            mca2 = MCA(n_components = n_components)
        eigen_values = np.loadtxt("fanalysis/tests/mca_eig.txt",
                                  delimiter=" ", dtype=float)
        n_components = self._adjust_n_components(n_components, eigen_values)
        X = np.loadtxt("fanalysis/tests/mca_row_coord.txt", delimiter=" ",
                       dtype=float)[:, :n_components]
        if data_type == "string":
            df = pd.read_table("fanalysis/tests/mca_data_string.txt",
                               header=0, index_col=0, delimiter="\t")
        else :
            df = pd.read_table("fanalysis/tests/mca_data_numeric.txt",
                               header=0, index_col=0, delimiter="\t")
        M = df.as_matrix()

        mca1.fit(M)
        Y_temp_1 = mca1.transform(M)
        Y1 = self._compute_Y(X, Y_temp_1, "row_coord_")
        assert_array_almost_equal(X, Y1)

        Y_temp_2 = mca2.fit_transform(M)
        Y2 = self._compute_Y(X, Y_temp_2, "row_coord_")
        assert_array_almost_equal(X, Y2)    
    
    def _X_Y_comparison(self, attr, test_file, data_type, n_components=None):
        """ This function compares the fitted values with the
        R FactoMiner output
        """
        mca = self._fit(data_type, n_components)
        eigen_values = np.loadtxt("fanalysis/tests/mca_eig.txt",
                                  delimiter=" ", dtype=float)
        n_components = self._adjust_n_components(n_components, eigen_values)
        X = np.loadtxt("fanalysis/tests/" + test_file, delimiter=" ",
                       dtype=float)[:, :n_components]
        Y_temp = getattr(mca, attr)
        Y = self._compute_Y(X, Y_temp, attr)
        assert_array_almost_equal(X, Y)

    def _row_topandas_comparison(self, data_type, n_components=None,
                                 row_labels=False):
        """ This function compares the output of the row_topandas method
        with the R FactoMiner output
        """
        if data_type == "string":
            df = pd.read_table("fanalysis/tests/mca_data_string.txt",
                               header=0, index_col=0, delimiter="\t")
        else :
            df = pd.read_table("fanalysis/tests/mca_data_numeric.txt",
                               header=0, index_col=0, delimiter="\t")
        M = df.as_matrix()
        if row_labels == False:
            labels = ["row" + str(x) for x in np.arange(0, M.shape[0])]
            mca = MCA(n_components = n_components, row_labels = None)
        else:
            labels = np.loadtxt("fanalysis/tests/mca_row_labels.txt",
                                delimiter=" ", dtype=str)
            mca = MCA(n_components = n_components, row_labels = labels)
        df_Y = mca.fit(M).row_topandas()
        Y = df_Y.as_matrix()
        df_Y_index = df_Y.index.values
        Y_row_coord_temp = Y[:, :mca.n_components_]
        
        eigen_values = np.loadtxt("fanalysis/tests/mca_eig.txt",
                                  delimiter=" ", dtype=float)
        n_components = self._adjust_n_components(n_components, eigen_values)
        X_row_coord = np.loadtxt("fanalysis/tests/mca_row_coord.txt",
                                 delimiter=" ", dtype=float)[:, :n_components]
        X_row_contrib = np.loadtxt("fanalysis/tests/mca_row_contrib.txt",
                                   delimiter=" ",
                                   dtype=float)[:, :n_components]
        X_row_cos2 = np.loadtxt("fanalysis/tests/mca_row_cos2.txt",
                                delimiter=" ", dtype=float)[:, :n_components]
        X = np.c_[X_row_coord, X_row_contrib, X_row_cos2]
        
        # test for data
        Y_row_coord = self._compute_Y(X_row_coord, Y_row_coord_temp,
                                      "row_coord_")
        Y[:, :mca.n_components_] = Y_row_coord
        assert_array_almost_equal(X, Y)

        # test for row_labels
        assert_array_equal(labels, df_Y_index)
        
    def _col_topandas_comparison(self, data_type, n_components=None,
                                 var_labels=False):
        """ This function compares the output of the col_topandas method
        with the R FactoMiner output
        """
        if data_type == "string":
            df = pd.read_table("fanalysis/tests/mca_data_string.txt",
                               header=0, index_col=0, delimiter="\t")
            with open("fanalysis/tests/mca_col_labels_string.txt", "r") \
                     as f:
                labels = f.read().split(" ")
        else :
            df = pd.read_table("fanalysis/tests/mca_data_numeric.txt",
                               header=0, index_col=0, delimiter="\t")
            with open("fanalysis/tests/mca_col_labels_numeric.txt", "r") \
                     as f:
                labels = f.read().split(" ")
        M = df.as_matrix()
        if var_labels == False:
            mca = MCA(n_components = n_components, var_labels = None)
        else:
            mca = MCA(n_components = n_components, var_labels = labels)
        df_Y = mca.fit(M).col_topandas()
        Y = df_Y.as_matrix()
        df_Y_index = df_Y.index.values
        Y_col_coord_temp = Y[:, :mca.n_components_]
        
        eigen_values = np.loadtxt("fanalysis/tests/mca_eig.txt",
                                  delimiter=" ", dtype=float)
        n_components = self._adjust_n_components(n_components, eigen_values)
        X_col_coord = np.loadtxt("fanalysis/tests/mca_col_coord.txt",
                                 delimiter=" ", dtype=float)[:, :n_components]
        X_col_contrib = np.loadtxt("fanalysis/tests/mca_col_contrib.txt",
                                   delimiter=" ",
                                   dtype=float)[:, :n_components]
        X_col_cos2 = np.loadtxt("fanalysis/tests/mca_col_cos2.txt",
                                delimiter=" ", dtype=float)[:, :n_components]
        X = np.c_[X_col_coord, X_col_contrib, X_col_cos2]
        
        # test for data
        Y_col_coord = self._compute_Y(X_col_coord, Y_col_coord_temp,
        "col_coord_")
        Y[:, :mca.n_components_] = Y_col_coord
        assert_array_almost_equal(X, Y)

        # test for col_labels
        assert_array_equal(labels, df_Y_index)

