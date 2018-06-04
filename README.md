# fanalysis

fanalysis is a Python module for Factorial Analysis distributed under the 3-Clause BSD license.

With this fanalysis package, you can perform:
- Simple Correspondence Analysis
- Multiple Correspondence Analysis
- Principal Components Analysis

Those statistical methods can be used in two ways:
- as descriptive methods ("datamining approach")
- as reduction methods in scikit-learn pipelines ("machine learning approach")

## Installation

### Dependencies

fanalysis requires:

    Python 3
    NumPy >= 1.11.0
    Matplotlib >= 2.0.0
    Scikit-learn >= 0.18.0
    Pandas >= 0.19.0

### User installation

You can install fanalysis using pip:

    pip install fanalysis

## Running the tests

After installation, you can launch the test suite from outside the source directory:

    python -m unittest

The philosophy of the unit tests consists in comparing the outputs of fanalysis (with various combinations of parameters) with the outputs of the R FactoMineR package.

## Documentation

The docstring is written in english.

Tutorials are available in french:

    https://github.com/OlivierGarciaDev/fanalysis/blob/master/doc/ca_tutorial.ipynb
    https://github.com/OlivierGarciaDev/fanalysis/blob/master/doc/mca_tutorial.ipynb
    https://github.com/OlivierGarciaDev/fanalysis/blob/master/doc/pca_tutorial.ipynb

## Author

Olivier Garcia (o.garcia.dev@gmail.com)
