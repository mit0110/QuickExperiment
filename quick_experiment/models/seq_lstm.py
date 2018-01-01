"""Model for sequence prediction with a LSTM RNN."""
import numpy
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from quick_experiment.models.lstm import LSTMModel


# Copied from tensorflow github.
def safe_div(numerator, denominator, name="value"):
    """Computes a safe divide which returns 0 if the denominator is zero.
    Note that the function contains an additional conditional check that is
    necessary for avoiding situations where the loss is zero causing NaNs to
    creep into the gradient computation.
    Args:
    numerator: An arbitrary `Tensor`.
    denominator: A `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
    Returns:
    The element-wise value of the numerator divided by the denominator.
    """
    return array_ops.where(
        math_ops.greater(denominator, 0),
        math_ops.div(numerator, array_ops.where(
            math_ops.equal(denominator, 0),
            array_ops.ones_like(denominator), denominator)),
        array_ops.zeros_like(numerator),
        name=name)


class SeqLSTMModel(LSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence.
    """

    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 logs_dirname='.', log_values=True,
                 max_num_steps=30, dropout_ratio=0.3, **kwargs):
        super(SeqLSTMModel, self).__init__(
            dataset, name, hidden_layer_size=hidden_layer_size,
            batch_size=batch_size, dropout_ratio=dropout_ratio,
            logs_dirname=logs_dirname, log_values=log_values,
            max_num_steps=max_num_steps, **kwargs)
        self.output_size = self.dataset.classes_num()

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.float32,
            (None, self.max_num_steps, self.dataset.feature_vector_size),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type,
            (None, self.max_num_steps, self.dataset.classes_num()),
            name='labels_placeholder')

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        input = self._build_input_layers()
        output = self._build_recurrent_layer(input)
        # outputs is a Tensor shaped
        # [batch_size, max_num_steps, hidden_size].

        # The last layer is for the classifier
        output = tf.reshape(output, [-1, self.hidden_layer_size])
        # Adding dropout
        output = tf.layers.dropout(inputs=output,
                                   rate=self.dropout_placeholder)
        logits = tf.layers.dense(
            output, self.output_size,
            kernel_initializer=tf.uniform_unit_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
            use_bias=True)
        logits = tf.reshape(
            logits, [-1, int(self.max_num_steps), int(self.output_size)])
        # logits is now a tensor [batch_size, max_num_steps, classes_num]
        return logits

    def _build_input_layers(self):
        """Applies a dropout to the input instances"""
        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')
        if self.dropout_ratio != 0:
            return tf.layers.dropout(inputs=self.instances_placeholder,
                                     rate=self.dropout_placeholder)
        return self.instances_placeholder

    def _build_recurrent_layer(self, input_op):
        # The recurrent layer
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(
            self.hidden_layer_size, forget_bias=1.0)
        with tf.name_scope('recurrent_layer') as scope:
            # Get the initial state. States will be a LSTMStateTuples.
            state_variable = self._build_state_variables(rnn_cell)
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, new_state = tf.nn.dynamic_rnn(
                rnn_cell, inputs=input_op,
                sequence_length=self.lengths_placeholder, scope=scope,
                initial_state=state_variable)
            # Define the state operations. This wont execute now.
            self.last_state_op = self._get_state_update_op(state_variable,
                                                           new_state)
            self.reset_state_op = self._get_state_update_op(
                state_variable,
                rnn_cell.zero_state(self.batch_size, tf.float32))
        return outputs

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]
        """
        mask = tf.sequence_mask(self.lengths_placeholder, self.max_num_steps)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder)
        loss = tf.div(
            tf.reduce_sum(tf.boolean_mask(loss, mask)),
            tf.cast(tf.reduce_sum(self.lengths_placeholder), loss.dtype))
        return loss

    def run_train_op(self, epoch, loss, partition_name, train_op):
        # Reset the neural_network_value
        self.sess.run(self.reset_state_op)
        # We need to run the train op cutting the sequence in chunks of
        # max_steps size
        loss_value = []
        feed_dict = None
        for feed_dict in self._fill_feed_dict(partition_name):
            feed_dict[self.dropout_placeholder] = self.dropout_ratio
            result = self.sess.run(
                [train_op, self.last_state_op, loss], feed_dict=feed_dict)
            loss_value.append(result[2])
        if self.logs_dirname is not None and epoch % 10 is 0 and feed_dict:
            self.write_summary(epoch, feed_dict)
        return numpy.mean(loss_value)

    def _get_step_dict(self, instances, labels, lengths, start_index,
                       step_size):
        step_instances = instances[:, start_index: start_index + step_size]
        step_labels = labels[:, start_index: start_index + step_size]
        step_lengths = numpy.clip(
            numpy.where(lengths >= start_index, lengths - start_index, 0),
            a_max=self.max_num_steps, a_min=0)
        feed_dict = {
            self.instances_placeholder: step_instances,
            self.lengths_placeholder: step_lengths,
            self.labels_placeholder: step_labels
        }
        return feed_dict

    def _build_predictions(self, logits):
        """Return a tensor with the predicted class for each instance.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                NUM_CLASSES].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps].
        """
        # Variable to store the predictions. The prediction for each element
        # is a vector with the most likely next
        # observed element in the sequence.
        return tf.argmax(logits, -1, name='batch_predictions')

    def _build_evaluation(self, logits):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                feature_vector + 1].
        Returns:
            A scalar int32 tensor with the number of examples (out of
            batch_size) that were predicted correctly.
        """
        predictions = self._build_predictions(logits)
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_performance'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                predictions, tf.argmax(self.labels_placeholder, -1),
                weights=mask)

        return accuracy, accuracy_update

    def _get_step_predictions(self, batch_prediction, batch_true, feed_dict):
        step_prediction = self.sess.run(self.predictions, feed_dict=feed_dict)
        labels = numpy.argmax(feed_dict[self.labels_placeholder], axis=-1)
        for index, length in enumerate(feed_dict[self.lengths_placeholder]):
            batch_prediction[index] = numpy.append(
                batch_prediction[index], step_prediction[index, :length])
            batch_true[index] = numpy.append(
                batch_true[index], labels[index, :length])

    def predict(self, partition_name, limit=-1):
        predictions = []
        true = []
        self.dataset.reset_batch()
        with self.graph.as_default():
            while (self.dataset.has_next_batch(self.batch_size, partition_name)
                   and (limit <= 0 or len(predictions) < limit)):
                batch_prediction = [numpy.array([]) for
                                    _ in range(self.batch_size)]
                batch_true = [numpy.array([]) for _ in range(self.batch_size)]
                for feed_dict in self._fill_feed_dict(partition_name,
                                                      reshuffle=False):
                    self._get_step_predictions(batch_prediction, batch_true,
                                               feed_dict)
                predictions.extend(batch_prediction)
                true.extend(batch_true)

        return numpy.array(true), numpy.array(predictions)
