

class Experiment(object):
    """Abstraction to train and evaluate of a model on a dataset.

    To create particular experiments, this class reads a configuration file
    with the specified parameters.

    Attributes:
        dataset (:obj: BaseDataset): An instance of BaseDataset (or
            subclass).
        name (string, optinal): A string with the experiment name. It will be
            used to identify the experiment files.
        results_dirpath (string, optional): A string with the name of the
            directory where to store the experiment.
        model (:obj:): The model to train or evaluate. It must have a
            scikit-learn compatible API with methods fit and predict.
    """

    def __init__(self, dataset, name=None, results_dirpath=None, config={}):
        self.dataset = dataset
        self.name = name
        self.results_dirpath = results_dirpath
        self._read_config(config)

    def _read_config(self, config):
        model_class = config.get('model', None)
        if model_class is None or not isinstance(model_class, type):
            raise ValueError('No model provided or model is not a class.')
        model_params = config.get('model_params', {})
        self.model = model_class(**model_params)

    def run(self):
        """Executes the experiment."""
        # Train classifier
        self.model.fit(self.dataset.datasets['train'].instances,)
        # Save model
        # Save the results


class SampledExperiment(Experiment):
    """Abstraction to run the same experiment with a sampled dataset."""
    pass
