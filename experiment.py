import os

import logging

import numpy
import pandas

logging.basicConfig(level=logging.INFO)

from sklearn import metrics


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
        if results_dirpath is not None:
            self.results_dirpath = results_dirpath
        else:
            self.results_dirpath = '.'
        self._read_config(config)

    def _read_config(self, config):
        self.model_class = config.get('model', None)
        if self.model_class is None or not isinstance(self.model_class, type):
            raise ValueError('No model provided or model is not a class.')
        self.model_arguments = config.get('model_arguments', {})

    def _create_model(self):
        self.model = self.model_class(self.dataset, **self.model_arguments)

    def _save_predictions(self, predictions, partition_name):
        if self.name is not None:
            filename = '{}_predictions_{}.csv'.format(partition_name, self.name)
        else:
            filename = '{}_predictions.csv'.format(partition_name)
        predictions = pandas.DataFrame(predictions, columns=['prediction'])
        predictions['true'] = self.dataset.datasets[partition_name].labels
        predictions.to_csv(os.path.join(self.results_dirpath, filename))

    def _get_metrics(self, predictions, partition_name):
        """Returns personalized metrics using predictions."""
        metric_values = numpy.array(metrics.precision_recall_fscore_support(
            self.dataset.datasets[partition_name].labels, predictions)).T
        report = (
            '\tPrecision\tRecall\tF1 Score\tSupport\n' +
            '\n'.join(['Class {}\t'.format(index) + '\t'.join(
                [str(value) for value in row])
                for index, row in enumerate(metric_values)]))
        logging.info('\n' + report)

    def run(self, save_model=False, save_predictions=False):
        """Executes the experiment.

        Args:
            save_model (bool): If true, saves the model in self.results_dirpath.
            save_predictions (bool): If true, saves test predictions in
                self.results_dirpath.
        """
        # Train classifier
        self._create_model()
        self.model.fit('train')
        # Save model
        if save_model:
            self.model.save_to_file(directory_name=self.results_dirpath,
                                    name=self.name)
        # Evaluate
        predictions = self.model.predict('test')
        self._get_metrics(predictions, partition_name='test')

        # Save the results
        if save_predictions:
            self._save_predictions(predictions, partition_name='test')


class SampledExperiment(Experiment):
    """Abstraction to run the same experiment with a sampled dataset."""

    def _get_name_for_sample(self, sample):
        if self.name is None:
            return 'sample' + str(sample)
        return self.name + '_sample' + str(sample)

    def _save_predictions(self, predictions, partition_name, sample=0):
        filename = '{}_predictions_{}.csv'.format(
            partition_name, self._get_name_for_sample(sample))
        predictions = pandas.DataFrame(predictions,
                                       columns=['prediction', 'true'])
        predictions.to_csv(os.path.join(self.results_dirpath, filename))

    def _get_metrics(self, predictions, partition_name=None):
        metric_values = []
        for true, prediction in predictions:
            metric_values.append(metrics.precision_recall_fscore_support(
                true, prediction, average='micro'
            )[:-1])
        metric_values = numpy.array(metric_values)
        report = ('\n\tPrecision\tRecall\tF1 Score\n' + 'mean\t' +
            '\t'.join([str(x) for x in metric_values.mean(axis=0)]) +
            '\nstd\t' + '\t'.join([str(x) for x in metric_values.std(axis=0)])
        )
        logging.info(report)

    def run(self, save_model=False, save_predictions=False):
        """Executes the experiment.

        Args:
            save_model (bool): If true, saves the trained models in
                self.results_dirpath.
            save_predictions (bool): If true, saves test predictions in
                self.results_dirpath.
        """
        predictions = []
        for sample in range(self.dataset.samples_num):
            # Train classifier
            self.dataset.set_current_sample(sample)
            self._create_model()
            self.model.fit('train')
            # Save model
            if save_model:
                self.model.save_to_file(directory_name=self.results_dirpath,
                                        name=self._get_name_for_sample(sample))
            # Evaluate
            predictions.append((self.dataset.datasets['test'].labels,
                                self.model.predict('test')))

            # Save the results
            if save_predictions:
                self._save_predictions(predictions[-1], partition_name='test',
                                       sample=sample)

        self._get_metrics(predictions)
