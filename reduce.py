import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class pca_experiment():

    ##
    # @param ndarray features | a numpy array of num_records X num_features
    # @param float variance_retained | use .95 to .99 range for best compression quality
    # @param str output_dir | No trailing slash, best to use absolute path to output fit data, dumps, etc.
    ##
    def __init__(self, features, file_name, variance_retained = .95, output_dir = '.'):
        # load the data
        self.features = features
        self.scaler = StandardScaler()
        self.pca = PCA(variance_retained)
        self.file_name = file_name
        self.output_dir = output_dir
    
    ##
    # Loading the various metrics from preprocessing of the training dataset
    #
    def preprocess_load(self):
        ndarray_metrics = ['mean_', 'var_', 'scale_']
        singular_value_metrics = {'n_samples_seen_': 'int'}
        return self.metrics_load('preprocess', ndarray_metrics, singular_value_metrics)

    ##
    # Saving all the attributes from the scaler for later use
    # (this should be called after fitting the data)
    # @return bool - If the files are successfully saved
    def preprocess_save(self):
        ndarray_metrics = [
            'mean_', 
            'var_', 
            'scale_', 
        ]
        
        singular_metrics = [
            'n_samples_seen_',        
        ]

        return self.metrics_save('preprocessor', ndarray_metrics, singular_metrics)

    ## 
    # Perform data preprocessing: specifically mean normalization and feature scaling
    # @param str op | Whether we are just fitting the data (to get mean, var, etc), perform data transformation or both
    # @return bool | if we are just fitting the data 
    # or ndarray transformed | The preprocessed features after applying preprocessing
    #
    def preprocess(self, op = 'fit'):
        if op == 'fit':
            # Fitting the data, getting mean/variance for future use.
            self.scaler.fit(self.features)
            self.preprocess_save()
            return True
        else:
            metrics = self.preprocess_load()
            # Either just perform transform with the mean/variance or fit and transform 
            if len(metrics['mean_']) > 1:
                print('Found mean and variance, perform preprocessing transform')
                for metric in metrics.keys():
                    # setting mean_, var_, scale_, n_samples_seen_
                    setattr(self.scaler, metric, metrics[metric])

                return self.scaler.transform(self.features)
            else:
                print('Mean and variance not found, perform preprocessing fit_transform')
                transformed = self.scaler.fit_transform(self.features)
                self.preprocess_save()              
                return transformed


    ##
    # Saving metrics (attributes) from the standard scaler or the PCA model
    #
    def metrics_save(self, model = 'preprocessor', ndarray_metrics = [], singular_metrics = [], format='%.8f', ):
        if model == 'preprocessor':
            object = self.scaler
            dir = 'preprocess'
        else:
            object = self.pca
            dir = 'pca'

        try:
            #splitting the base filename from the extension 
            basename = self.file_name.split('.')[0]
            basedir = f'{self.output_dir}/pca/{basename}/{dir}'
            os.makedirs(basedir, exist_ok=True)

            # Saving numpy arrays
            for ndarray_metric in ndarray_metrics:
                np.savetxt(fname=f'{basedir}/{ndarray_metric}.txt', X=getattr(object, ndarray_metric), fmt=format, delimiter=',')                

            # Saving singular values
            for singular_metric in singular_metrics:
                with open(f'{basedir}/{singular_metric}.txt', 'w') as filehandle:
                    filehandle.write(str(getattr(object, singular_metric)))

        except AttributeError:
            return False         

    ## 
    # @param str dir | preprocess or pca
    # @return dict metrics | a dictionary keyed by the metrics (mean, var, scale, n_samples_seen) 
    # that can be set to the scaler to perform transform subsquently to the entire data set (train + test)
    # for information on the data type for each keyed data
    # see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #
    def metrics_load(self, dir = 'preprocess', ndarray_metrics = [], singular_metrics = {}):
        metrics = {}
        basename = self.file_name.split('.')[0]

        # Loading all the ndarray 
        for file in ndarray_metrics:
            try:
                fh = open(f"{self.output_dir}/pca/{basename}/{dir}/{file}.txt", "r")
                data = []
                for line in fh:
                    features = line.split(',')
                    if len(features) == 1:
                        features = line

                    data.append(features)

                if bool(data) == True:
                    data = np.array(data) 
                    data = data.astype(float)

                metrics.update({file: data})
                fh.close()
            except FileNotFoundError:
                metrics.update({file: []})

        # Loading all the singular values
        for file in singular_metrics.keys():
            try:
                fh = open(f"{self.output_dir}/pca/{basename}/{dir}/{file}.txt", "r")
                data = fh.read()
                # Some data casting
                if singular_metrics[file] == 'float':
                    data = float(data)
                else:
                    data = int(data)
                metrics.update({file: data})
            except FileNotFoundError:
                metrics.update({file: []})

        return metrics

    ##
    # Saving all attributes of the PCA model after fitting the data
    #
    def pca_save(self):

        ndarray_metrics = [
            'components_', 
            'explained_variance_', 
            'explained_variance_ratio_', 
            'singular_values_', 
            'mean_', 
        ]
        
        singular_metrics = [
            'n_components_', 
            'n_features_', 
            'n_samples_', 
            'noise_variance_'           
        ]

        return self.metrics_save('pca', ndarray_metrics, singular_metrics)

    ##
    # Load metrics from the fitted PCA model
    #
    def pca_load(self):
        ndarray_metrics = [
            'components_', 
            'explained_variance_', 
            'explained_variance_ratio_', 
            'singular_values_', 
            'mean_', 
        ]

        singular_metrics = {
            'n_components_': 'int', 
            'n_features_': 'int', 
            'n_samples_': 'int', 
            'noise_variance_': 'float'              
        }

        return self.metrics_load('pca', ndarray_metrics, singular_metrics)

    ## 
    # Perform dimensionality reduction with PCA 
    #
    def decompose(self, op = 'fit'):
        preprocessed_features = self.preprocess('transform')

        if op == 'fit':
            self.pca.fit(preprocessed_features)
            self.pca_save()
            return True
        else:
            metrics = self.pca_load()
            if (len(metrics['mean_']) > 0):
                print('Found prefitted PCA model attributes, transform the data only')

                # Setting the attributes
                for metric in metrics.keys():
                    # setting mean_, components_, etc
                    setattr(self.pca, metric, metrics[metric])               

                reduced_features = self.pca.transform(preprocessed_features)
            else:
                print('Prefitted PCA model attributes not found, fit and transform the data')
                reduced_features = self.pca.fit_transform(preprocessed_features)
                self.pca_save()
            
            return reduced_features
        
    ## 
    # Dumping the reduced feature set to a CSV (without labels)
    # @param string format | default format of 7 decmial places with expoential notation
    #
    def dump(self, format='%1.7g'):
        # Reducing the data set to  to M x K using PCA
        reduced_features = self.decompose('transform')
        np.savetxt(fname=f'{self.output_dir}/{self.file_name}', X=reduced_features, fmt=format, delimiter=',')
