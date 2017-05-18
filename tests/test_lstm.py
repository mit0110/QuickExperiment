
from quick_experiment import dataset
import numpy
import random
import tensorflow as tf
import unittest

from sklearn.preprocessing import OneHotEncoder
from quick_experiment.models import lstm


class LSTMModelTest(unittest.TestCase):
    """Tests for building and running a LSTMModel instance"""

    def setUp(self):
        tf.reset_default_graph()
        num_examples = 100
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of two elements.
        self.matrix = [
            numpy.array([numpy.array([x, x+1]) for x in range(random.randint(3, 20))])
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
            'log_values': 0, 'training_epochs': 10}

    def test_build_network(self):
        """Test if the LSTMModel is correctly built."""
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)
        model.fit(close_session=True)

    def test_predict(self):
        """Test if the LSTMModel is correctly built."""
        # Check build does not raise errors
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)
        model.fit()
        true, predictions = model.predict('test')
        expected_size = ((self.dataset.num_examples('test') /
                          model.batch_size) * model.batch_size)
        self.assertEqual(true.shape[0], expected_size)
        self.assertEqual(true.shape, predictions.shape)

    def test_reshape_output(self):
        """Test if the output are correctly reshaped after the dynamic_rnn call.
        """
        batch_size = self.model_arguments['batch_size']
        max_num_steps = 20
        hidden_size = self.model_arguments['hidden_layer_size']
        model = lstm.LSTMModel(self.dataset, **self.model_arguments)

        lengths_array = numpy.random.random_integers(
            max_num_steps / 2, max_num_steps, batch_size)
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
            max_num_steps / 2, max_num_steps*3, batch_size)
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


class SeqPredictionModelTest(unittest.TestCase):
    """Tests for building and running a SeqPredictionModel instance"""

    def _get_one_hot_encoding(self, x):
        result = numpy.zeros(self.feature_size)
        result[x] = 1
        return result

    def _get_random_sequence(self):
        return numpy.array([
            self._get_one_hot_encoding(x % self.feature_size)
            for x in range(random.randint(3, self.max_num_steps))])

    def setUp(self):
        num_examples = 50
        self.max_num_steps = 10
        self.feature_size = 5
        self.batch_size = 5  # Validation size
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of two elements.
        self.matrix = [self._get_random_sequence() for _ in range(num_examples)]
        self.matrix = numpy.array(self.matrix)
        self.partition_sizes = {
            'train': 0.65, 'test': 0.25, 'validation': 0.1
        }
        self.dataset = dataset.UnlabeledSequenceDataset()
        self.dataset.create_samples(self.matrix, None, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)

        self.model_arguments = {
            'hidden_layer_size': 40, 'batch_size': self.batch_size,
            'logs_dirname': None,
            'log_values': 10, 'training_epochs': 100,
            'max_num_steps': self.max_num_steps}
        # Check build does not raise errors
        self.model = lstm.SeqPredictionModel(self.dataset,
                                             **self.model_arguments)

    def test_build_network(self):
        """Test if the Seq2SeqLSTMModel is correctly built."""
        self.model.fit(close_session=True)

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

    def test_build_loss(self):
        """Test if the loss (binary cross entropy) is built correctly."""
        with tf.Graph().as_default():
            # outputs is a Tensor shaped
            # [batch_size, max_num_steps, hidden_size].
            logits = tf.random_uniform(
                (self.model_arguments['batch_size'], self.max_num_steps,
                 self.dataset.classes_num('train')))
            self.model._build_inputs()
            loss = self.model._build_loss(logits)
            sigmoid = tf.nn.sigmoid(logits)
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            feed_dict = self.model._fill_feed_dict('train').next()
            sigmoid, result = sess.run([sigmoid, loss], feed_dict=feed_dict)
            sess.close()
        # Calculate loss
        expected_loss = []
        for index, sequence in enumerate(
                feed_dict[self.model.labels_placeholder]):
            sequence_loss = []
            for element_index in range(
                    feed_dict[self.model.lengths_placeholder][index]):
                correct_label = sequence[element_index]
                sequence_loss.append(-1 * numpy.log(numpy.sum(
                    sigmoid[index, element_index] * correct_label)))
            expected_loss.append(numpy.mean(sequence_loss))
        expected_loss = numpy.mean(expected_loss)

        self.assertAlmostEqual(expected_loss, result, places=6)

    def _get_correctly_predicted(self, labels, lengths, logit_labels):
        total_correct = 0
        for sequence_index, sequence in enumerate(labels):
            for label_index, true_label_vector in enumerate(
                    sequence[:lengths[sequence_index]]):
                true_label = numpy.argmax(true_label_vector)
                if true_label == logit_labels[sequence_index, label_index]:
                    total_correct += 1
        return total_correct

    def _run_evaluation(self, iterations=1):
        """Returns true accuracy and all accuracy_op results."""
        total_correct = 0
        total_evaluated = 0
        with tf.Graph().as_default():
            logits = tf.random_uniform(
                (self.model_arguments['batch_size'], self.max_num_steps,
                 self.dataset.classes_num('train')))
            self.model._build_inputs()
            accuracy_op, accuracy_update_op = self.model._build_evaluation(
                logits)
            sess = tf.Session()
            sess.run(tf.local_variables_initializer())
            for iteration in range(iterations):
                feed_dict = self.model._fill_feed_dict('train').next()
                logits_ev, _ = sess.run([logits, accuracy_update_op],
                                                   feed_dict=feed_dict)
                labels = feed_dict[self.model.labels_placeholder]
                lengths = feed_dict[self.model.lengths_placeholder]
                logit_labels = numpy.argmax(logits_ev, axis=-1)
                total_correct += self._get_correctly_predicted(labels, lengths,
                                                               logit_labels)
                total_evaluated += lengths.sum()
            final_accuracy = sess.run([accuracy_op])[0]
            sess.close()
        return final_accuracy, total_correct / float(total_evaluated)

    def test_build_evaluation(self):
        accuracy, true_accuracy = self._run_evaluation(1)
        self.assertIsInstance(accuracy, numpy.float32)
        self.assertAlmostEqual(true_accuracy, accuracy, places=5)

    def test_build_evaluation_accuracy(self):
        epochs = 2
        accuracy, true_accuracy = self._run_evaluation(epochs)
        self.assertAlmostEqual(true_accuracy, accuracy, places=5)

    def test_build_predictions(self):
        with tf.Graph().as_default():
            logits = tf.random_uniform(
                (self.model_arguments['batch_size'], self.max_num_steps,
                 self.model_arguments['hidden_layer_size']))
            transform_op = self.model._build_predictions(logits)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                logits, predictions = sess.run([logits, transform_op])

        self.assertEqual(logits.shape[:-1], predictions.shape)
        self.assertEqual(predictions.shape, (self.model_arguments['batch_size'],
                                             self.max_num_steps))
        max_arguments = numpy.argmax(logits, axis=-1)
        self.assertTrue(numpy.array_equal(max_arguments, predictions))

    def test_predict(self):
        """Test the prediction for each sequence element is the probability
        of the next element in sequence, for all possible elements."""
        self.model.fit(close_session=False)
        predictions, true = self.model.predict('test')
        self.assertIsInstance(predictions, numpy.ndarray)
        for true_sequence, predicted_sequence in zip(true, predictions):
            self.assertEqual(true_sequence.shape[0],
                             predicted_sequence.shape[0])


if __name__ == '__main__':
    unittest.main()
