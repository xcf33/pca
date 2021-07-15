import os, pathlib
import numpy as np
from reduce import pca_experiment

'''
Testing the PCA algorithm on a training set
'''
class test_train():

    def __init__(self):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir_path = pathlib.Path(config_dir)
        app_dir = config_dir_path.parent
        self.app_dir = app_dir
        self.ids, features = self.load('train.txt')
        self.variance_retained = .99
        self.pca = pca_experiment(features, 'train.txt', self.variance_retained, f'{self.app_dir}/output')

    def run(self):
        print(f"Training set shape: {self.pca.features.shape[0]},{self.pca.features.shape[1]}")
        print(f"Number of components before PCA: {self.pca.features.shape[1]}")
        self.pca.dump()
        self.dump_labels()
        print(f"Running PCA with a variance retained at {self.variance_retained}")
        print(f"Number of components after PCA: {self.pca.pca.n_components_}")
        print(f"Compressed data set is saved in the output directory")


    def dump_labels(self):
        with open(f"{self.app_dir}/output/train_label.txt", "w") as filehandle:
            ids = list(map(lambda x: x + '\n', self.ids))
            filehandle.writelines(ids)

    ##
    # The data file should be a id|feature1,feature2,feature3 format file 
    # return a list of ids and a numpy array of the features
    #
    def load(self, file_name):
        features = []
        unique_ids = []
        try:
            fh = open(f"{self.app_dir}/tests/{file_name}", "r")
            for line in fh:      
                id, _feature = str(line).split('|')
                feature = _feature.split(',')
                if feature[0] != 'NULL':
                    unique_ids.append(id)
                    features.append(feature)
            fh.close()

            if bool(features) == True:
                features = np.array(features)
                features = features.astype(float)

            return [unique_ids, features]

        except FileNotFoundError:
            return [False, False]

'''
Testing the PCA algorithm on a test set data (1 sample)
'''
class test_test(test_train):
    def __init__(self):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir_path = pathlib.Path(config_dir)
        self.app_dir = config_dir_path.parent
        self.ids, features = self.load('test.txt')
        self.variance_retained = .99
        # We are passing in the train.txt to look for the fitted metrics from the training set
        self.pca = pca_experiment(features, 'train.txt', self.variance_retained, f'{self.app_dir}/output')

    def run(self):
        print(f"Performing PCA on a test set of shape: {self.pca.features.shape[0]},{self.pca.features.shape[1]}")
        reduced_features = self.pca.decompose('transform')
        print('Shape after performing PCA')
        print(reduced_features.shape)
        print('Dump')
        print(reduced_features)
