import io, os, json, redis, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class pca_experiment():

    ##
    # @param ndarray features | a numpy array of num_records X num_features
    # @param float variance_retained | use .95 to .99 range for best compression quality
    # @param str storage | file or redis, what do we use to store the preprocessing metrics and reduced features, defaults to file
    # @param dict storage_config | associated configuration when choosing file or storage
    ##
    def __init__(self, features, variance_retained = .95, storage = 'file', storage_config = {}):
        # load the data
        self.features = features
        self.scaler = StandardScaler()
        self.pca = PCA(variance_retained)

        '''
        file storage config looks like
        {
            'name': some_unique_id_name,
            'output_dir': '/path/to/output',
        }

        redis storage config looks like
        {
            'host': 127.0.0.1,
            'port': 6379,
            'key': some_unique_id_name,
            'encoding': 'utf-8',
        }
        '''
        self.storage_config = storage_config

        # Get an instance of redis going if possible
        if storage == 'redis':
            try:
                self.redis = redis.Redis(host=storage_config['host'], port=storage_config['port'], db=0)
                self.REDIS_ENCODING = storage_config['encoding']
            except (TimeoutError, ConnectionError):
                self.redis = False
        else:
            self.redis = False

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
    # Saving the preprocess metrics (mean, std, etc) as well as PCA processing metrics 
    # Based on the storage configuration, this can be saved to Redis or file (by default)
    # @param (str) model | preprocessor or pca, used to fetch the associated attributes for saving
    # @param (list) ndarray_metrics, the list of metrics that should be saved as numpy array (from both preprocessor and pca)
    # @param (list) singular metrics, the list of metrics that should be saved as singular values (from both preprocessor and pca)
    #
    def metrics_save(self, model = 'preprocessor', ndarray_metrics = [], singular_metrics = [], format='%.8f'):
        if bool(self.redis) == True:
            # Do redis Save
            print('Saving metrics to redis')
            self.metrics_save_redis(model, ndarray_metrics, singular_metrics)
        else:
            # Save the metrics to file
            print('Saving metrics to file')
            self.metrics_save_file(model, ndarray_metrics, singular_metrics, format)


    ##
    # Saving the preprocessing and PCA metrics to redis, see @metrics_save
    # This method allows for centralized loading of the metrics to just transform the data
    #
    def metrics_save_redis(self, model = 'preprocessor', ndarray_metrics = [], singular_metrics = []):
        if model == 'preprocessor':
            object = self.scaler
            prefix = 'preprocess'
        else:
            object = self.pca
            prefix = 'pca'

        # Data dictionary to save
        data = {}

        try:
            for ndarray_metric in ndarray_metrics:
                # A way for us to serialize numpy array without type casting
                memfile = io.BytesIO()
                np.save(memfile, getattr(object, ndarray_metric))
                memfile.seek(0)
                data[ndarray_metric] = json.dumps(memfile.read().decode('latin-1'))

            for singular_metric in singular_metrics:
                data[singular_metric] = getattr(object, singular_metric)

            # Save all metrics in Redis in one pipeline
            pipe = self.redis.pipeline()
            for metric in data.keys():
                # The cache key will look like {8_client}_{pca}_{mean_}
                cache_key = f"{self.storage_config['key']}_{prefix}_{metric}"
                pipe.set(cache_key, data[metric])
            pipe.execute()
        except AttributeError:
            return False

    ##
    # Saving the preprocessing and PCA metrics to file, see @metrics_save
    #
    def metrics_save_file(self, model = 'preprocessor', ndarray_metrics = [], singular_metrics = [], format='%.8f'):
        if model == 'preprocessor':
            object = self.scaler
            dir = 'preprocess'
        else:
            object = self.pca
            dir = 'pca'

        try:
            #splitting the base filename from the extension 
            basedir = f"{self.storage_config['output_dir']}/pca/{self.storage_config['name']}/{dir}"
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
    # @param str prefix | preprocess or pca
    # @return dict metrics | a dictionary keyed by the metrics (mean, var, scale, n_samples_seen) 
    # that can be set to the scaler to perform transform subsquently to the entire data set (train + test)
    # for information on the data type for each keyed data
    # see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #
    def metrics_load(self, prefix = 'preprocess', ndarray_metrics = [], singular_metrics = {}):
        metrics = {}
        if bool(self.redis) == True:
            print(f'loading {prefix} metrics from Redis')
            metrics = self.metrics_load_redis(prefix, ndarray_metrics, singular_metrics)
        else:
            print(f'loading {prefix} metrics from File')
            metrics = self.metrics_load_file(prefix, ndarray_metrics, singular_metrics)

        return metrics

    ##
    # Loading metrics from Redis
    #
    def metrics_load_redis(self, prefix = 'preprocess', ndarray_metrics = [], singular_metrics = {}):
        metrics = {}

        for metric in ndarray_metrics:
            # Not using redis pipeline for simplicity and data granularity 
            cache_key = f"{self.storage_config['key']}_{prefix}_{metric}"

            metric_value = self.redis.get(cache_key)
            if metric_value == None:
                metrics[metric] = []
            else:
                memfile = io.BytesIO()
                memfile.write(json.loads(metric_value.decode(self.REDIS_ENCODING)).encode('latin-1'))
                memfile.seek(0)
                ndarray_value = np.load(memfile, allow_pickle=True)
                metrics[metric] = ndarray_value

        # Singular metrics is a keyed dictionary with value of the data type (e.g. float/int)
        for metric in singular_metrics.keys():
            cache_key = f"{self.storage_config['key']}_{prefix}_{metric}"

            metric_value = self.redis.get(cache_key)
            if metric_value == None:
                metrics[metric] = []
            else:
                if singular_metrics[metric] == 'float':
                    metrics[metric] = float(metric_value.decode(self.REDIS_ENCODING))
                else:
                    metrics[metric] = int(metric_value.decode(self.REDIS_ENCODING))

        return metrics

    ##
    # Load metrics from file, see @metrics_load
    #
    def metrics_load_file(self, dir = 'preprocess', ndarray_metrics = [], singular_metrics = {}):
        metrics = {}

        # Loading all the ndarray 
        for file in ndarray_metrics:
            try:
                fh = open(f"{self.storage_config['output_dir']}/pca/{self.storage_config['name']}/{dir}/{file}.txt", "r")
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
                fh = open(f"{self.storage_config['output_dir']}/pca/{self.storage_config['name']}/{dir}/{file}.txt", "r")
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
            metrics = self.trained_metrics()
            if metrics != False:
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
    # Fetch the trained metrics
    #
    def trained_metrics(self):
        metrics = self.pca_load()
        if (len(metrics['mean_']) > 0):
            return metrics
        else:
            return False

    ## 
    # Dumping the reduced feature set to a CSV (without labels)
    # @param string format | default format of 7 decmial places with expoential notation
    #
    def dump(self, format='%1.7g'):
        # Reducing the data set to  to M x K using PCA
        reduced_features = self.decompose('transform')
        np.savetxt(fname=f"{self.storage_config['output_dir']}/{self.storage_config['name']}.csv", X=reduced_features, fmt=format, delimiter=',')
