#Principle Component Analysis (PCA)

This is a python implementation of the PCA algorithm for performing dimensionality reduction. 

## Prerequisite

You will need to install numpy and sklearn to use this package, for example:

`pip install sklearn, numpy`

## Advantage of using this repo

This repo will save preprocessing and PCA fitting information from the training set data to file so when you perform PCA on your data set it will be faster subsequently.

## Testing/Demo

Simply run this in your terminal in the repo directory

```
python ./test.py
```

It will take a test file of a 56 x 4096 dataset and perform PCA at a variance retained of .99. The result should be a dataset with a dimension of 56 x 52