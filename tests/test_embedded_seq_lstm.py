import numpy
import random
import tensorflow as tf
import unittest

from quick_experiment import dataset
from quick_experiment.models import embedded_seq_lstm


class EmbeddedSeqLSTMModelTest(unittest.TestCase):
    """Tests for building and running a SeqPredictionModel instance"""

    def _get_random_sequence(self):
        return numpy.array([
            x % self.feature_size
            for x in range(random.randint(3, self.max_num_steps))])

    def _get_random_labels(self, matrix):
        result = []
        for x in matrix:
            if random.random() > 0.5:
                result.append((x+1) * -1)
            else:
                result.append(x + 1)
        return numpy.array(result)

    def setUp(self):
        num_examples = 50
        self.max_num_steps = 10
        self.feature_size = 5
        self.batch_size = 5  # Validation size
        # The matrix is an array of sequences of varying sizes.
        self.matrix = [self._get_random_sequence() for _ in range(num_examples)]
        self.labels = self._get_random_labels(self.matrix)
        self.matrix = numpy.array(self.matrix)
        self.partition_sizes = {
            'train': 0.65, 'test': 0.25, 'validation': 0.1
        }
        self.dataset = dataset.EmbeddedSequenceDataset()
        self.dataset.create_samples(self.matrix, self.labels, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)

        self.model_arguments = {
            'hidden_layer_size': 40, 'batch_size': self.batch_size,
            'logs_dirname': None,
            'log_values': 10, 'training_epochs': 50,
            'max_num_steps': self.max_num_steps,
            'embedding_size': 7}
        # Check build does not raise errors
        self.model = embedded_seq_lstm.EmbeddedSeqLSTMModel(
            self.dataset, **self.model_arguments)

    def test_build_network(self):
        """Test if the SeqLSTMModels is correctly built."""
        self.model.fit(close_session=True)

    def test_single_distributed_layer(self):
        """Test the model uses the same weights for the time distributed layer.
        """
        self.model.fit()
        with self.model.graph.as_default():
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='softmax_layer'):
                self.assertNotIn('softmax_layer-1', variable.name,
                                 msg='More than one layer created.')

    def test_fit_loss(self):
        # Emulate the first part of the fit call
        with tf.Graph().as_default():
            self.model._build_inputs()
            # Build a Graph that computes predictions from the inference model.
            logits = self.model._build_layers()
            # Add to the Graph the Ops for loss calculation.
            loss = self.model._build_loss(logits)
            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self.model._build_train_operation(loss)

            init = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.model.sess = tf.Session()
            self.model.sess.run([init, init_local])
            for epoch in range(10):
                loss_value = self.model.run_train_op(epoch, loss, 'train',
                                                     train_op)
                self.assertFalse(numpy.isnan(loss_value),
                                 msg='The loss value is nan.')

    def test_predict(self):
        """Test if the SeqLSTMModels is correctly built."""
        self.model.fit(close_session=False)
        true, prediction = self.model.predict('test')
        self.assertEqual(true.shape, prediction.shape)
        self.assertEqual(numpy.max([x.max() for x in true]), 1)
        self.assertEqual(numpy.min([x.min() for x in true]), 0)
        self.assertLessEqual(numpy.max([x.max() for x in prediction]), 1)
        self.assertGreaterEqual(numpy.min([x.min() for x in prediction]), 0)


if __name__ == '__main__':
    unittest.main()
