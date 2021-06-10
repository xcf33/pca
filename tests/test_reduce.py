import os, pathlib
import numpy as np
from reduce import pca_experiment

class test_reduce():

    def __init__(self):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir_path = pathlib.Path(config_dir)
        app_dir = config_dir_path.parent
        self.app_dir = app_dir
        self.ids, features = self.load('test.txt')
        self.variance_retained = .99
        self.pca = pca_experiment(features, 'test.txt', self.variance_retained, f'{self.app_dir}/output')

    def run(self):
        print(f"Number of components before PCA: {self.pca.features.shape[1]}")
        self.pca.dump()
        self.dump_labels()
        print(f"Running PCA with a variance retained at {self.variance_retained}")
        print(f"Number of components after PCA: {self.pca.pca.n_components_}")
        print(f"Compressed data set is saved in the output directory")


    def dump_labels(self):
        with open(f"{self.app_dir}/output/test_label.txt", "w") as filehandle:
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
            fh = open(f"{self.app_dir}/tests/test.txt", "r")
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