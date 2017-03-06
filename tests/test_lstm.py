import dataset
import numpy
import random
import tensorflow as tf
import unittest
from models import lstm

class LSTMModelTest(unittest.TestCase):
    """Tests for building and running a LSTMModel instance"""

    def setUp(self):
        num_examples = 100
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of two elements.
        self.matrix = [
            [numpy.array([x, x+1]) for x in range(random.randint(3, 20))]
            for _ in range(num_examples)]
        self.matrix = numpy.array(self.matrix)
        self.labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int32)
        self.partition_sizes = {
            'train': 0.65, 'test': 0.25, 'validation': 0.1
        }
        self.dataset = dataset.SequenceDataset()
        self.dataset.create_samples(self.matrix, self.labels, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)

    def test_build_network(self):
        """Test if the LSTMModel is correctly built."""
        model_arguments = {'hidden_layer_size': 50, 'batch_size': 20,
                            'logs_dirname': None,
                            'log_values': True, 'training_epochs': 100}
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **model_arguments)
        model.fit(close_session=True)

    def test_reshape_output(self):
        """Test if the output are correctly reshaped after the dynamic_rnn call.
        """
        batch_size = 100
        max_num_steps = 20
        hidden_size = 50
        model_arguments = {'hidden_layer_size': hidden_size,
                           'logs_dirname': None, 'batch_size': batch_size,
                           'log_values': True, 'training_epochs': 100}
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **model_arguments)

        with tf.Graph().as_default():
            # outputs is a Tensor shaped
            # [batch_size, max_num_steps, hidden_size].
            outputs = tf.random_uniform(
                (batch_size, max_num_steps, hidden_size))
            transform_output_op = model.reshape_output(outputs)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                old_output, new_output = sess.run(
                    [outputs, transform_output_op])
                self.assertEqual(new_output.shape, (batch_size, hidden_size))
                # Check instance by instance the output is correct
                for index, row in enumerate(old_output):
                    self.assertEqual(row.shape, (max_num_steps, hidden_size))
                    self.assertTrue(numpy.array_equal(
                        row[-1,:], new_output[index]))

if __name__ == '__main__':
    unittest.main()
