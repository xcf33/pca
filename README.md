# Principle Component Analysis (PCA)

This is a python implementation of the PCA algorithm for performing dimensionality reduction. 

## Prerequisite

You will need to install numpy and sklearn to use this package, for example:

`pip install sklearn, numpy`

## Why use this repo

This repo will save the preprocessing and PCA fitting information from the training set data to file or Redis so when you perform PCA on your data set it will be faster subsequently.

## Example code

Perform PCA and storing all preprocessing and PCA fitting information on file. The example uses a data set of 100 randomly generated samples of 500 features.

```
from reduce import pca_experiment
import numpy as np

features = np.random.rand(100, 500)
variance_retained = .99
storage_config = {
    'output_dir': 'output',
    'name': 'sample-data-set'
}
pca = pca_experiment(features, variance_retained, 'file', storage_config)
reduced_features = pca.decompose('transform')

print(reduced_features.shape)
```

The above example will perform PCA on the training set data as well as reduce the dimension of the training set `pca.decompose('transform')`. It will store all trained weights and other fitting metrics to the output directory which allows you to easily run PCA on new test data in this data set. For example:

```
features = np.random.rand(1, 500)
variance_retained = .99
storage_config = {
    'output_dir': 'output',
    'name': 'sample-data-set'
}
pca = pca_experiment(features, variance_retained, 'file', storage_config)
reduced_features = pca.decompose('transform')
```

Make sure the `name` element in the dictionary is the same as the one you used in the training set data.

It is also possible to use Redis to store the preprocessing information and the PCA fitting data. You simply pass the corresponding storage configurations. Redis allows you to store the information in a central host for distributed code processing.

```
from reduce import pca_experiment
import numpy as np, redis

features = np.random.rand(100, 500)
variance_retained = .99
storage_config = {
    'encoding': 'utf-8',
    'key': 'sample-data-set',
    'host': '127.0.0.1',
    'port': 6379,
}
pca = pca_experiment(features, variance_retained, 'redis', storage_config)
reduced_features = pca.decompose('transform')

print(reduced_features.shape)
```



## Testing/Demo

Simply run this in your terminal in the repo directory

```
python ./test.py
```

It will take a test file of a 56 x 4096 dataset and perform PCA at a variance retained of .99. The result should be a dataset with a dimension of 56 x 52