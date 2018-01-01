
import numpy
import random
import tensorflow as tf
import unittest

from quick_experiment import dataset
from quick_experiment.models import lstm


class LSTMModelTest(unittest.TestCase):
    """Tests for building and running a LSTMModel instance"""

    def setUp(self):
        tf.reset_default_graph()
        num_examples = 100
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of two elements.
        self.max_num_steps = 10
        self.matrix = [numpy.array([numpy.array([x, x+1])
                                    for x in range(random.randint(
                                        3, 2*self.max_num_steps))])
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
        self.model_arguments = {
            'hidden_layer_size': 50, 'batch_size': 20, 'logs_dirname': None,
            'log_values': 0}

    def test_build_network(self):
        """Test if the LSTMModel is correctly built."""
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)
        model.fit(close_session=True)

    def test_predict(self):
        """Test if the LSTMModel returns consistent predictions."""
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)
        model.fit()
        true, predictions = model.predict('test')
        expected_size = ((self.dataset.num_examples('test') //
                          model.batch_size) * model.batch_size)
        self.assertEqual(true.shape[0], expected_size)
        self.assertEqual(true.shape, predictions.shape)
        self.assertGreaterEqual(predictions.min(), 0)

    def test_evaluate(self):
        """Test if the LSTMModel returns a valid accuracy value."""
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)
        model.fit()
        metric = model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)

    def test_reshape_output(self):
        """Test if the output are correctly reshaped after the dynamic_rnn call.
        """
        batch_size = self.model_arguments['batch_size']
        max_num_steps = 20
        hidden_size = self.model_arguments['hidden_layer_size']
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)

        lengths_array = numpy.random.random_integers(
            max_num_steps // 2, max_num_steps, batch_size)
        new_output, old_output = self._transform_output(lengths_array,
                                                        max_num_steps, model)
        self.assertEqual(new_output.shape, (batch_size, hidden_size))
        # Check instance by instance the output is correct
        for index, row in enumerate(old_output):
            self.assertEqual(row.shape, (max_num_steps, hidden_size))
            self.assertTrue(numpy.array_equal(
                row[lengths_array[index]-1, :], new_output[index]))

    def test_reshape_output_big_sequences(self):
        """Test if the output are correctly reshaped after the dynamic_rnn call.

        The sequences have bigger size than max_num_steps
        """
        batch_size = self.model_arguments['batch_size']
        max_num_steps = 20
        hidden_size = self.model_arguments['hidden_layer_size']
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)

        lengths_array = numpy.random.random_integers(
            max_num_steps // 2, max_num_steps*3, batch_size)
        new_output, old_output = self._transform_output(lengths_array,
                                                        max_num_steps, model)
        self.assertEqual(new_output.shape, (batch_size, hidden_size))
        # Check instance by instance the output is correct
        for index, row in enumerate(old_output):
            self.assertEqual(row.shape, (max_num_steps, hidden_size))
            relevent_output_index = min(lengths_array[index],
                                        max_num_steps)
            self.assertTrue(numpy.array_equal(
                row[relevent_output_index - 1, :], new_output[index]))

    def _transform_output(self, lengths_array, max_num_steps, model):
        with tf.Graph().as_default():
            # outputs is a Tensor shaped
            # [batch_size, max_num_steps, hidden_size].
            outputs = tf.random_uniform(
                (self.model_arguments['batch_size'], max_num_steps,
                 self.model_arguments['hidden_layer_size']))
            lengths = tf.constant(lengths_array, dtype=tf.int32)
            transform_op = model.reshape_output(outputs, lengths)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                old_output, new_output = sess.run([outputs, transform_op])
        return new_output, old_output

    def test_fill_feed_dict(self):
        batch_size = self.model_arguments['batch_size']
        for instance in self.dataset._instances:
            self.assertLessEqual(instance.shape[0], 2*self.max_num_steps)
        model = lstm.LSTMModel(self.dataset, max_num_steps=self.max_num_steps,
                               **self.model_arguments)
        model.build_all()
        batch_iterator = model._fill_feed_dict(partition_name='train')
        instances = next(batch_iterator)[model.instances_placeholder]
        self.assertEqual(instances.shape, (batch_size, self.max_num_steps,
                                           self.dataset.feature_vector_size))
        # As the maximum sequence lenght is 2, this should run exactly two times
        instances = next(batch_iterator)[model.instances_placeholder]
        self.assertEqual(instances.shape, (batch_size, self.max_num_steps,
                                           self.dataset.feature_vector_size))
        with self.assertRaises(StopIteration):
            next(batch_iterator)


if __name__ == '__main__':
    unittest.main()
