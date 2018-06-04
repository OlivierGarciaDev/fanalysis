# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fanalysis",
    version="0.0.1",
    author="OlivierGarciaDev",
    author_email="o.garcia.dev@gmail.com",
    description="Python module for Factorial Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.11.0",
                      "matplotlib>=2.0.0",
                      "scikit-learn>=0.18.0",
                      "pandas>=0.19.0"],
    python_requires=">=3",
    package_data={"": ["*.txt"]},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
