#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import find_packages, setup

# to install project, go to main project directory and
# run `python setup.py install`
setup(
    name='src',
    packages=find_packages(),
    version='0.0.1',
    author="Mélisande Teng",
    author_email="tengmeli@mila.quebec",
    description="A package for height estimation using GSV images",
    python_requires='>=3.6'
)