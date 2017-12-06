import os

import utils


class BaseModel(object):
    """Abstraction to represent trainable ML models using Dataset instances.

    Attributes:
        dataset: An instance of BaseDataset

    Args:
        dataset (:obj: BaseDataset): An instance of BaseDataset (or
            subclass).
        **kwargs: Additional arguments.
    """

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.graph = None

    def fit(self, partition_name):
        """Train the model with a dataset partition.

        Args:
            partition_name (str): The partition of the dataset to use for
                fitting.
        """
        raise NotImplementedError

    def predict(self, partition_name):
        """Return the prediction over a dataset partition.

        Args:
            partition_name (str): The partition of the dataset to use for
                prediction.
        """
        raise NotImplementedError

    def save_to_file(self, directory_name, name=None):
        """Save model to directory_name.

        Args:
            directory_name (string): Name of directory to save files.
            name (string, optional): additional name to add into the dataset
                files.
        """
        raise NotImplementedError


class SkleanrModel(BaseModel):
    """Wrapper around scikit-learn models compatible with BaseModel API."""

    def __init__(self, dataset, model_class, sklearn_model_arguments={},
                 name=None):
        super(SkleanrModel, self).__init__(dataset)
        self.model = model_class(**sklearn_model_arguments)
        self.name = name

    def fit(self, partition_name):
        self.model.fit(self.dataset.datasets[partition_name].instances,
                       self.dataset.datasets[partition_name].labels)

    def predict(self, partition_name):
        return (
            self.model.predict(self.dataset.datasets[partition_name].instances),
            self.dataset.datasets[partition_name].labels)

    def save_to_file(self, directory_name, name=None):
        if name is not None:
            filename = os.path.join(directory_name, '{}_model.p'.format(name))
        else:
            filename = os.path.join(directory_name, 'model.p')

        utils.safe_mkdir(directory_name)
        utils.pickle_to_file(self.model, filename)
