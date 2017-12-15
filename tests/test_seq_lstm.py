import numpy
import random
import tensorflow as tf
import unittest

from quick_experiment import dataset
from quick_experiment.models import seq_lstm


class SeqLSTMModelTest(unittest.TestCase):
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
            'log_values': 10,
            'max_num_steps': self.max_num_steps}
        # Check build does not raise errors
        tf.reset_default_graph()
        self.model = seq_lstm.SeqLSTMModel(self.dataset, **self.model_arguments)

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

    @staticmethod
    def _get_correctly_predicted(labels, lengths, logit_labels):
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
                feed_dict = next(self.model._fill_feed_dict('train'))
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
        true, predictions = self.model.predict('test')
        self.assertIsInstance(predictions, numpy.ndarray)
        for true_sequence, predicted_sequence in zip(true, predictions):
            self.assertEqual(true_sequence.shape[0],
                             predicted_sequence.shape[0])

    def test_evaluate(self):
        """Test if the model returns a valid accuracy value."""
        self.model.fit(close_session=False)
        metric = self.model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)


if __name__ == '__main__':
    unittest.main()
