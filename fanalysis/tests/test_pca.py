# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd

from fanalysis.pca import PCA


class TestPca(unittest.TestCase):
    """ Unit tests for the PCA class
    """
    def test_pca(self):
        """ Test for the fit_transfom operation - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            self._fit_transform_comparison(std_unit = su, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._fit_transform_comparison(std_unit = su, n_components = i)
    
    def test_eigen_(self):
        """ Test for the eigen values - Comparison with the R FactoMiner
        output
        """
        for su in [True, False]:
            pca_eig = "pca_eig_scale_unit_true.txt" if su \
                        else "pca_eig_scale_unit_false.txt"
            self._X_Y_comparison("eig_", pca_eig, std_unit = su,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("eig_", pca_eig, std_unit = su,
                                     n_components = i)
    
    def test_row_coord_(self):
        """ Test for the rows coordinates - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            pca_row_coord = "pca_row_coord_scale_unit_true.txt" if su \
                        else "pca_row_coord_scale_unit_false.txt"
            self._X_Y_comparison("row_coord_", pca_row_coord, std_unit = su,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("row_coord_", pca_row_coord,
                                     std_unit = su, n_components = i)

    def test_row_contrib_(self):
        """ Test for the rows contributions - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            pca_row_contrib = "pca_row_contrib_scale_unit_true.txt" if su \
                        else "pca_row_contrib_scale_unit_false.txt"
            self._X_Y_comparison("row_contrib_", pca_row_contrib,
                                 std_unit = su, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("row_contrib_", pca_row_contrib,
                                     std_unit = su, n_components = i)
    
    def test_row_cos2_(self):
        """ Test for the rows cos2 - Comparison with the R FactoMiner
        output
        """
        for su in [True, False]:
            pca_row_cos2 = "pca_row_cos2_scale_unit_true.txt" if su \
                        else "pca_row_cos2_scale_unit_false.txt"
            self._X_Y_comparison("row_cos2_", pca_row_cos2, std_unit = su,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("row_cos2_", pca_row_cos2, std_unit = su,
                                     n_components = i)
    
    def test_col_coord_(self):
        """ Test for the columns coordinates - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            pca_col_coord = "pca_col_coord_scale_unit_true.txt" if su \
                        else "pca_col_coord_scale_unit_false.txt"
            self._X_Y_comparison("col_coord_", pca_col_coord, std_unit = su,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_coord_", pca_col_coord,
                                     std_unit = su, n_components = i)

    def test_col_contrib_(self):
        """ Test for the columns contributions - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            pca_col_contrib = "pca_col_contrib_scale_unit_true.txt" if su \
                else "pca_col_contrib_scale_unit_false.txt"
            self._X_Y_comparison("col_contrib_", pca_col_contrib,
                                 std_unit = su, n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_contrib_", pca_col_contrib,
                                     std_unit = su, n_components = i)
    
    def test_col_cos2_(self):
        """ Test for the columns cos2 - Comparison with the R FactoMiner
        output
        """
        for su in [True, False]:
            pca_col_cos2 = "pca_col_cos2_scale_unit_true.txt" if su \
                else "pca_col_cos2_scale_unit_false.txt"
            self._X_Y_comparison("col_cos2_", pca_col_cos2, std_unit = su,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_cos2_", pca_col_cos2, std_unit = su,
                                     n_components = i)

    def test_col_cor_(self):
        """ Test for the columns correlations - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            pca_col_cor = "pca_col_cor_scale_unit_true.txt" if su \
                else "pca_col_cor_scale_unit_false.txt"
            self._X_Y_comparison("col_cor_", pca_col_cor, std_unit = su,
                                 n_components = None)
            for i in np.arange(-10, 10, 0.5):
                self._X_Y_comparison("col_cor_", pca_col_cor, std_unit = su,
                                     n_components = i)

    def test_row_topandas(self):
        """ Test for the row_topandas method - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            self._row_topandas_comparison(std_unit = su, n_components = None,
                                          row_labels = False)
            self._row_topandas_comparison(std_unit = su, n_components = None,
                                          row_labels = True)
            for i in np.arange(-10, 10, 0.5):
                self._row_topandas_comparison(std_unit = su, n_components = i,
                                              row_labels = False)
                self._row_topandas_comparison(std_unit = su, n_components = i,
                                              row_labels = True)
    
    def test_col_topandas(self):
        """ Test for the col_topandas method - Comparison with the
        R FactoMiner output
        """
        for su in [True, False]:
            self._col_topandas_comparison(std_unit = su, n_components = None,
                                          col_labels = False)
            self._col_topandas_comparison(std_unit = su, n_components = None,
                                          col_labels = True)
            for i in np.arange(-10, 10, 0.5):
                self._col_topandas_comparison(std_unit = su, n_components = i,
                                              col_labels = False)
                self._col_topandas_comparison(std_unit = su, n_components = i,
                                              col_labels = True)
    
    def _fit(self, std_unit, n_components):
        """ This function fits the model to the data """
        df = pd.read_table("fanalysis/tests/pca_data.txt", header=0,
                           index_col=0, delimiter="\t")
        M = df.as_matrix()
        pca = PCA(std_unit = std_unit, n_components = n_components)
        pca.fit(M)
        return pca

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
        if ((attr == "row_coord_")
                or (attr == "col_coord_")
                or (attr == "col_cor_")):
            x = X[1, :]
            y = Y_temp[1, :]
            z = x * y
            for i in np.arange(0, z.shape[0]):
                z[i] = 1 if z[i] >=0 else -1
            return Y_temp * z.reshape(1, -1)
            
        else:
            return Y_temp
    
    def _fit_transform_comparison(self, std_unit, n_components=None):
        """ This function compares the result of the fit_transform
        operation with the R FactoMineR output
        """
        if n_components is None:
            pca1 = PCA(std_unit = std_unit)
            pca2 = PCA(std_unit = std_unit)
        else:
            pca1 = PCA(std_unit = std_unit, n_components = n_components)
            pca2 = PCA(std_unit = std_unit, n_components = n_components)
        if std_unit:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_true.txt",
                                      delimiter=" ", dtype=float)
            n_components = self._adjust_n_components(n_components,
                                                     eigen_values)
            X = np.loadtxt("fanalysis/tests/" +
                           "pca_row_coord_scale_unit_true.txt",
                           delimiter=" ", dtype=float)[:, :n_components]
        else:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_false.txt",
                                      delimiter=" ", dtype=float)
            n_components = self._adjust_n_components(n_components,
                                                     eigen_values)
            X = np.loadtxt("fanalysis/tests/" +
                           "pca_row_coord_scale_unit_false.txt",
                           delimiter=" ", dtype=float)[:, :n_components]
        df = pd.read_table("fanalysis/tests/pca_data.txt", header=0,
                           index_col=0, delimiter="\t")
        M = df.as_matrix()

        pca1.fit(M)
        Y_temp_1 = pca1.transform(M)
        Y1 = self._compute_Y(X, Y_temp_1, "row_coord_")
        assert_array_almost_equal(X, Y1)

        Y_temp_2 = pca2.fit_transform(M)
        Y2 = self._compute_Y(X, Y_temp_2, "row_coord_")
        assert_array_almost_equal(X, Y2)    
    
    def _X_Y_comparison(self, attr, test_file, std_unit, n_components=None):
        """ This function compares the fitted values with the
        R FactoMiner output
        """
        pca = self._fit(std_unit, n_components)
        if std_unit:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_true.txt",
                                      delimiter=" ", dtype=float)
            #test_file += "scale_unit_true.txt"
        else:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_false.txt",
                                      delimiter=" ", dtype=float)
            #test_file += "scale_unit_false.txt"
        n_components = self._adjust_n_components(n_components, eigen_values)
        X = np.loadtxt("fanalysis/tests/" + test_file, delimiter=" ",
                       dtype=float)[:, :n_components]
        Y_temp = getattr(pca, attr)
        Y = self._compute_Y(X, Y_temp, attr)
        assert_array_almost_equal(X, Y)
    
    def _row_topandas_comparison(self, std_unit, n_components=None,
                                 row_labels=False):
        """ This function compares the output of the row_topandas method
        with the R FactoMiner output
        """
        df = pd.read_table("fanalysis/tests/pca_data.txt", header=0,
                           index_col=0, delimiter="\t")
        M = df.as_matrix()
        if row_labels == False:
            labels = ["row" + str(x) for x in np.arange(0, M.shape[0])]
            pca = PCA(std_unit = std_unit, n_components = n_components,
                      row_labels = None)
        else:
            labels = np.loadtxt("fanalysis/tests/pca_row_labels.txt",
                                delimiter=" ", dtype=str)
            pca = PCA(std_unit = std_unit, n_components = n_components,
                      row_labels = labels)
        df_Y = pca.fit(M).row_topandas()
        Y = df_Y.as_matrix()
        df_Y_index = df_Y.index.values
        Y_row_coord_temp = Y[:, :pca.n_components_]
        
        if std_unit:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_true.txt",
                                      delimiter=" ", dtype=float)
            n_components = self._adjust_n_components(n_components,
                                                     eigen_values)
            X_row_coord = np.loadtxt("fanalysis/tests/" +
                                     "pca_row_coord_scale_unit_true.txt",
                                     delimiter=" ",
                                     dtype=float)[:, :n_components]
            X_row_contrib = np.loadtxt("fanalysis/tests/" +
                                       "pca_row_contrib_scale_unit_true.txt",
                                       delimiter=" ",
                                       dtype=float)[:, :n_components]
            X_row_cos2 = np.loadtxt("fanalysis/tests/" +
                                    "pca_row_cos2_scale_unit_true.txt",
                                    delimiter=" ",
                                    dtype=float)[:, :n_components]
        else:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_false.txt",
                                      delimiter=" ", dtype=float)
            n_components = self._adjust_n_components(n_components,
                                                     eigen_values)
            X_row_coord = np.loadtxt("fanalysis/tests/" +
                                     "pca_row_coord_scale_unit_false.txt",
                                     delimiter=" ",
                                     dtype=float)[:, :n_components]
            X_row_contrib = np.loadtxt("fanalysis/tests/" +
                                       "pca_row_contrib_scale_unit_false.txt",
                                       delimiter=" ",
                                       dtype=float)[:, :n_components]
            X_row_cos2 = np.loadtxt("fanalysis/tests/" +
                                    "pca_row_cos2_scale_unit_false.txt",
                                    delimiter=" ",
                                    dtype=float)[:, :n_components]
        X = np.c_[X_row_coord, X_row_contrib, X_row_cos2]
        
        # test for data
        Y_row_coord = self._compute_Y(X_row_coord, Y_row_coord_temp,
                                      "row_coord_")
        Y[:, :pca.n_components_] = Y_row_coord
        assert_array_almost_equal(X, Y)

        # test for row_labels
        assert_array_equal(labels, df_Y_index)
        
    def _col_topandas_comparison(self, std_unit, n_components=None,
                                 col_labels=False):
        """ This function compares the output of the col_topandas method
        with the R FactoMiner output
        """
        df = pd.read_table("fanalysis/tests/pca_data.txt", header=0,
                           index_col=0, delimiter="\t")
        M = df.as_matrix()
        if col_labels == False:
            labels = ["col" + str(x) for x in np.arange(0, M.shape[1])]
            pca = PCA(std_unit = std_unit, n_components = n_components,
                      row_labels = None)
        else:
            labels = np.loadtxt("fanalysis/tests/pca_col_labels.txt",
                                delimiter=" ", dtype=str)
            pca = PCA(std_unit = std_unit, n_components = n_components,
                      col_labels = labels)
        df_Y = pca.fit(M).col_topandas()
        Y = df_Y.as_matrix()
        df_Y_index = df_Y.index.values
        Y_col_coord_temp = Y[:, :pca.n_components_]
        
        if std_unit:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_true.txt",
                                      delimiter=" ", dtype=float)
            n_components = self._adjust_n_components(n_components,
                                                     eigen_values)
            X_col_coord = np.loadtxt("fanalysis/tests/" +
                                     "pca_col_coord_scale_unit_true.txt",
                                     delimiter=" ",
                                     dtype=float)[:, :n_components]
            X_col_contrib = np.loadtxt("fanalysis/tests/" +
                                       "pca_col_contrib_scale_unit_true.txt",
                                       delimiter=" ",
                                       dtype=float)[:, :n_components]
            X_col_cos2 = np.loadtxt("fanalysis/tests/" +
                                    "pca_col_cos2_scale_unit_true.txt",
                                    delimiter=" ",
                                    dtype=float)[:, :n_components]
        else:
            eigen_values = np.loadtxt("fanalysis/tests/" +
                                      "pca_eig_scale_unit_false.txt",
                                      delimiter=" ", dtype=float)
            n_components = self._adjust_n_components(n_components,
                                                     eigen_values)
            X_col_coord = np.loadtxt("fanalysis/tests/" +
                                     "pca_col_coord_scale_unit_false.txt",
                                     delimiter=" ",
                                     dtype=float)[:, :n_components]
            X_col_contrib = np.loadtxt("fanalysis/tests/" +
                                       "pca_col_contrib_scale_unit_false.txt",
                                       delimiter=" ",
                                       dtype=float)[:, :n_components]
            X_col_cos2 = np.loadtxt("fanalysis/tests/" +
                                    "pca_col_cos2_scale_unit_false.txt",
                                    delimiter=" ",
                                    dtype=float)[:, :n_components]
        X = np.c_[X_col_coord, X_col_contrib, X_col_cos2]
        
        # test for data
        Y_col_coord = self._compute_Y(X_col_coord, Y_col_coord_temp,
                                      "col_coord_")
        Y[:, :pca.n_components_] = Y_col_coord
        assert_array_almost_equal(X, Y)

        # test for col_labels
        assert_array_equal(labels, df_Y_index)

