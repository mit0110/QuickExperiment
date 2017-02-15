import logging
import os

import numpy

import utils

from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit


Partition = namedtuple('Partition', ['instances', 'labels'])


class BaseDataset(object):
    """Abstraction to model generic datasets with samples and partitions.

    A dataset is a particular organization of a set of instances. The typical
    partitions are: training, validation and testing. To avoid redundance of
    information, partitions are represented as a matrix, a label vector and the
    set of indices for each partition.

    Args:

    Attributes:
        datasets (dict): a map from partition names to Partitions.
        indices (dict): a map from partition names to their indices.
    """

    def __init__(self):
        self.datasets = {}
        self.indices = {}

    def num_examples(self, partition_name='train'):
        """Returns the number of examples in a partition of the dataset.

        Args:
            partition_name (str): name of the partition to use.

        Returns:
            int: the number of examples.
        """
        raise NotImplementedError

    def save_to_files(self, directory_name):
        """Saves all dataset files into a directory.

        Args:
            directory_name (string): Name of directory to save files.
        """
        raise NotImplementedError

    def create_from_matrixes(self, matrix, indices, labels=None):
        """Creates the dataset from a matrix and the set of indices.

        Args:
            matrix (:obj: iterable): the iterable with instances.
            indices (dict): a map from partitions to their indices. The
                partition will be created filtering matrix with the given
                indices.
            labels (:obj: iterable, optional): if given, the partitions will
                include also a filtered selection of labels using the same
                indices.
        """
        raise NotImplementedError

    def load_from_files(self, directory_name, *args):
        """Builds dataset from files saved in the directory directory_name.

        Args:
            directory_name (string): Name of directory to read files from.
            *args: extra arguments.
        """
        raise NotImplementedError

    def log_matrixes_shapes(self):
        message = '\t'
        for partition in self.datasets.keys():
            message += ' {} num examples {}.'.format(
                partition, self.num_examples(partition))
        logging.info(message)

    def next_batch(self, batch_size, partition_name='train'):
        """Generates batches of instances and labels from a partition.

        If the size of the partition is exceeded, the partition and the labels
        are shuffled to generate further batches.

        Args:
            batch_size (int): the size of each batch to generate.
            partition_name (str): the name of the partition to create the
                batches from.
        """
        raise NotImplementedError

    @staticmethod
    def _get_objective_filename(directory_name, objective, name):
        if name:
            filename = '{}_{}.p'.format(objective, name)
        else:
            filename = '{}.p'.format(objective)
        return os.path.join(directory_name, filename)


class SimpleDataset(BaseDataset):
    """Simple dataset with 2D numpy array as instances.

    Attributes:
        datasets (dict): a map from partition names to Partitions.
        indices (dict): a map from partition names to their indices.
    """

    def __init__(self):
        super(SimpleDataset, self).__init__()
        self._iteration_index = 0

    def num_examples(self, partition_name='train'):
        """Returns the number of examples in a partition of the dataset.

        Args:
            partition_name (str): name of the partition to use.

        Returns:
            int: the number of examples.
        """
        return self.datasets[partition_name].instances.shape[0]

    def save_to_files(self, directory_name, name=None):
        """Saves all dataset files into a directory.

        Args:
            directory_name (string): Name of directory to save files.
            name (string, optional): additional name to add into the dataset
                files.
        """
        filename = self._get_objective_filename(directory_name, 'indices', name)
        utils.pickle_to_file(self.indices, filename)

    def create_from_matrixes(self, matrix, indices, labels=None):
        """Creates the dataset from a matrix and the set of indices.

        Args:
            matrix (:obj: iterable): the iterable with instances.
            indices (dict): a map from partitions to their indices. The
                partition will be created filtering matrix with the given
                indices.
            labels (:obj: iterable, optional): if given, the partitions will
                include also a filtered selection of labels using the same
                indices.
        """
        self.indices = indices
        for partition, index in indices.iteritems():
            partition_labels = labels[index] if labels is not None else None
            self.datasets[partition] = Partition(instances=matrix[index],
                                                 labels=partition_labels)

    def load_from_files(self, directory_name, instances_filename=None,
                        labels_filename=None, name=None):
        """Builds dataset from files saved in the directory directory_name.

        Args:
            directory_name (string): Name of directory to read files from.
            instances_filename (string): Name of the file containing the
                instances matrix in numpy compressed format.
            labels_filename (string, optional): Name of the file containing the
                labels in pickled format.
            name(string, optional): additional name to add into the dataset
                files.
        """
        instances = None
        if instances_filename:
            instances = numpy.load(instances_filename)
        if instances is None:
            logging.error('Error loading instances from file {}'.format(
                instances_filename
            ))
        labels = None
        if labels_filename:
            labels = utils.pickle_from_file(labels_filename)
        indices = utils.pickle_from_file(self._get_objective_filename(
            directory_name, 'indices', name))
        self.create_from_matrixes(instances, indices, labels)

    def next_batch(self, batch_size, partition_name='train'):
        """Generates batches of instances and labels from a partition.

        If the size of the partition is exceeded, the partition and the labels
        are shuffled to generate further batches.

        Args:
            batch_size (int): the size of each batch to generate.
            partition_name (str): the name of the partition to create the
                batches from.
        """
        start = self._iteration_index
        self._iteration_index += batch_size

        if self._iteration_index > self.num_examples(partition_name):
            # Shuffle the data
            perm = numpy.arange(self.num_examples())
            numpy.random.shuffle(perm)
            new_labels = None
            if self.datasets[partition_name].labels is not None:
                new_labels = self.datasets[partition_name].labels[perm]
            self.datasets[partition_name] = Partition(
                self.datasets[partition_name].instances[perm], new_labels)
            # Start next iteration
            start = 0
            self._iteration_index = batch_size
            assert batch_size <= self.num_examples(partition_name)

        end = self._iteration_index
        batch_labels = None
        if self.datasets[partition_name].labels is not None:
            batch_labels = self.datasets[partition_name].labels[start:end]
        return self.datasets[partition_name].instances[start:end], batch_labels


class BaseSampledDataset(BaseDataset):
    """Abstraction to handle a dataset divided into multiple samples.

    Attributes:
        datasets (dict): a map from partition names to Partitions.
        indices (dict): a map from partition names to their indices.
        samples_num (int): the number of samples created.
        current_sample (int): the number of the sample that is being currently
            load in the dataset attibute
    """

    def __init__(self):
        super(BaseSampledDataset, self).__init__()
        self.samples_num = None
        self.current_sample = None
        self._sample_indices = []

    @property
    def indices(self):
        return self._sample_indices[self.current_sample]

    def create_samples(self, instances, labels, samples_num, partition_sizes):
        """Generates the partitions for each sample.

        Args:
            instances (:obj: iterable): instances to divide in samples.
            labels (:obj: iterable): labels to divide in samples.
            samples_num (int): the number of samples to create.
            partition_sizes (dict): a map from the partition names to their
                proportional sizes. The sum of all values must be less or equal
                to one.
        """
        raise NotImplementedError

    def _load_sample(self):
        raise NotImplementedError

    def set_current_sample(self, sample):
        """Changes the dataset to the current sample."""
        assert sample < self.samples_num and sample >= 0
        self.current_sample = sample
        self._load_sample()

    def save_to_files(self, directory_name, name=None):
        """Saves all the sample files into the directory directory_name.

        Args:
            directory_name (string): Name of directory to save files.
            name (string, optional): additional name to add into the dataset
                files.
        """
        utils.safe_mkdir(directory_name)
        super(BaseSampledDataset, self).save_to_files(directory_name)
        utils.pickle_to_file(self._sample_indices, self._get_objective_filename(
            directory_name, 'sample_indices', name))


class SimpleSampledDataset(BaseSampledDataset):
    """Simple sampled dataset using 2D numpy arrays as instances (in memory).

    Attributes:
        datasets (dict): a map from partition names to Partitions.
        indices (dict): a map from partition names to their indices.
        samples_num (int): the number of samples created.
        current_sample (int): the number of the sample that is being currently
            load in the dataset attibute.
    """

    def __init__(self):
        super(SimpleSampledDataset, self).__init__()
        # Numpy matrixes can be loaded in memory
        self._instances = None
        self._labels = None

    def create_samples(self, instances, labels, samples_num, partition_sizes):
        """Creates samples with a Kfold partition generator."""
        assert sum(partition_sizes.values()) <= 1.0
        assert instances.shape[0] == labels.shape[0]
        self.samples_num = samples_num
        self.sample_indices = [
            dict.fromkeys(partition_sizes) for _ in range(samples_num)]
        self._instances = instances
        self._labels = labels
        remainings = [range(instances.shape[0])] * samples_num
        for partition, size in reversed(sorted(
                partition_sizes, key=partition_sizes.get)):
            # Recalculate the proportion of instances to take with the new
            # size of the remaining instances.

            logging.info("New remaining size {}".shape(size))
            self._get_partition_indices(partition, remainings, size)

    def _get_partition_indices(self, partition, remainings, size):
        size = int(self._instances.shape[0] * size / remainings[0].shape[0])
        partition_generator = StratifiedShuffleSplit(n_splits=self.samples_num,
                                                     test_size=size)
        for sample in range(self.samples_num):
            sample_remainings = remainings[sample]
            split = partition_generator.split(
                self._instances[sample_remainings],
                self._labels[sample_remainings])
            remainings[sample], self.sample_indices[sample][partition] = split


    def load_from_files(self, directory_name, instances_filename=None,
                        labels_filename=None, name=None):
        """Load sample from the directory directory_name.

        Args:
            directory_name (string): Name of directory to read files from.
            instances_filename (string): Name of the file containing the
                instances matrix in numpy compressed format.
            labels_filename (string, optional): Name of the file containing the
                labels in pickled format.
            name(string, optional): additional name to add into the dataset
                files.
        """
        super(BaseSampledDataset, self).load_from_files(directory_name)
        if instances_filename:
            self._instances = numpy.load(instances_filename)
        if self._instances is None:
            logging.error('Error loading instances from file {}'.format(
                instances_filename
            ))
        if labels_filename:
            self._labels = utils.pickle_from_file(labels_filename)

        self._sample_indices = utils.pickle_from_file(
            self._get_objective_filename(directory_name,
                                         'sample_indices', name))
        self.samples_num = len(self._sample_indices)

    def _load_sample(self):
        """Loads current sample in the attibute datasets."""
        indices = self._sample_indices[self.current_sample]
