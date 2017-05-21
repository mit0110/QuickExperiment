"""Model for sequence prediction with a LSTM RNN."""
import numpy
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from quick_experiment.models.lstm import LSTMModel


# Copied from tensorflow github.
def _safe_div(numerator, denominator, name="value"):
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


def time_distributed(incoming, fn, args=None):
    """Applies fn to all elements of incoming along the second dimension."""
    timestep = incoming.shape[1]
    x = tf.unstack(incoming, axis=1)

    # Create the first set of variables
    with tf.variable_scope('time_distributed') as scope:
        result = [fn(x[0], scope=scope, **args)]
    # Share the variables with the following layers
    with tf.variable_scope('time_distributed', reuse=True) as scope:
        result.extend([fn(x[i], scope=scope, reuse=True, **args)
                       for i in range(1, timestep)])
    try:
        x = map(lambda t: tf.reshape(
            t, [-1, 1]+t.get_shape().as_list()[1:]), result)
    except:
        x = list(map(lambda t: tf.reshape(
            t, [-1, 1]+t.get_shape().as_list()[1:]), result))
    return tf.concat(x, 1)


class SeqLSTMModel(LSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence.
    """

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

    def reshape_output(self, outputs, lengths):
        return outputs

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        # The recurrent layer
        output = self._build_recurrent_layer()
        # outputs is a Tensor shaped
        # [batch_size, max_num_steps, hidden_size].

        # The last layer is for the classifier
        fully_connected_args = [
            self.dataset.classes_num(), 'linear', True,
            'truncated_normal', 'zeros', # weights and bias initializer
            'L2', # regularizer
            0.001, True, True,
            False,  # Reuse
        ]
        layer_args = {
            'num_outputs': self.dataset.classes_num(),
        }
        logits = time_distributed(output, tf.contrib.layers.fully_connected,
                                  layer_args)
        # logits is now a tensor [batch_size, max_num_steps, classes_num]
        return logits

    def _build_state_variables(self, cell):
        # Get the initial state and make a variable out of it
        # to enable updating its value.
        state_c, state_h = cell.zero_state(self.batch_size, tf.float32)
        return (tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))

    @staticmethod
    def _get_state_update_op(state_variables, new_state):
        # Add an operation to update the train states with the last state
        # Assign the new state to the state variables on this layer
        return (state_variables[0].assign(new_state[0]),
                state_variables[1].assign(new_state[1]))

    def _build_recurrent_layer(self):
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
                rnn_cell, inputs=self.instances_placeholder,
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
        logits_shape = logits.get_shape()
        assert self.labels_placeholder.get_shape()[1] == logits_shape[1]
        # Calculate the cross entropy for the prediction of each step.
        # The (binary) cross entropy is defined as
        # -\sum_{x \in {a, b}} p(x) log(q(x))
        # We want to compare the true label against the predicted probability
        # of that label
        logits = tf.log(tf.nn.sigmoid(logits))

        cross_entropy = -tf.multiply(
            logits, tf.cast(self.labels_placeholder, dtype=logits.dtype))
        # cross_entropy has shape [batch_size, max_num_steps, classes_num] but
        # has only one active record per element of each sequence
        # Now we sum over all classes.
        cross_entropy = tf.reduce_sum(cross_entropy, axis=2)
        cross_entropy = tf.multiply(cross_entropy, tf.sequence_mask(
            self.lengths_placeholder, logits_shape[1], dtype=logits.dtype))
        # cross_entropy has shape [batch_size, max_num_steps]

        # Remove the elements that are not part of the sequence.
        # We take the average cross entropy of each sequence, taking into
        # consideration the lenght of the sequence.
        cross_entropy = _safe_div(tf.reduce_sum(cross_entropy, axis=1),
                                  tf.to_float(self.lengths_placeholder))
        # cross_entropy now has shape [batch_size, ]

        # Finally, we take the average over all examples in batch.
        return tf.reduce_mean(cross_entropy)

    def run_train_op(self, epoch, loss, partition_name, train_op):
        # Reset the neural_network_value
        self.sess.run(self.reset_state_op)
        # We need to run the train op cutting the sequence in chunks of
        # max_steps size
        loss_value = []
        for feed_dict in self._fill_feed_dict(partition_name):
            result = self.sess.run(
                [train_op, self.last_state_op, loss], feed_dict=feed_dict)
            loss_value.append(result[2])
        if self.logs_dirname is not None and epoch % 10 is 0:
            self.write_summary(epoch, feed_dict)
        return numpy.mean(loss_value)

    def _fill_feed_dict(self, partition_name='train', reshuffle=True):
        """Fills the feed_dict for training the given step.

        Args:
            partition_name (string): The name of the partition to get the batch
                from.

        Yields:
            The feed dictionaries mapping from placeholders to values with
                each chunk of size self.max_num_steps for the current batch.
        """
        batch = self.dataset.next_batch(
            self.batch_size, partition_name, pad_sequences=True,
            max_sequence_length=self.max_num_steps, reshuffle=reshuffle)
        if batch is None:
            raise ValueError('Batch is None')
        instances, labels, lengths = batch
        step_size = self.max_num_steps or instances.shape[1]
        assert instances.shape[1] % step_size == 0
        for start_index in range(0, instances.shape[1], step_size):
            feed_dict = self._get_step_dict(instances, labels, lengths,
                                            start_index, step_size)
            yield feed_dict

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

    def _get_step_predictions(self, batch_prediction, batch_true, feed_dict):
        step_prediction = self.sess.run(self.predictions, feed_dict=feed_dict)
        labels = numpy.argmax(feed_dict[self.labels_placeholder], axis=-1)
        for index, length in enumerate(feed_dict[self.lengths_placeholder]):
            batch_prediction[index] = numpy.append(
                batch_prediction[index], step_prediction[index, :length])
            batch_true[index] = numpy.append(
                batch_true[index], labels[index, :length])

    def predict(self, partition_name):
        predictions = []
        true = []
        self.dataset.reset_batch()
        with self.graph.as_default():
            while self.dataset.has_next_batch(self.batch_size, partition_name):
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
        with tf.name_scope('evaluation_accuracy'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                predictions, tf.argmax(self.labels_placeholder, -1),
                weights=mask)

        return accuracy, accuracy_update

    def evaluate_validation(self, correct_predictions):
        partition = 'validation'
        # Reset the accuracy variables
        stream_vars = [i for i in tf.local_variables()
                       if i.name.split('/')[0] == 'evaluation_accuracy']
        accuracy_op, accuracy_update_op = correct_predictions
        self.dataset.reset_batch()
        accuracy = None
        while self.dataset.has_next_batch(self.batch_size, partition):
            for feed_dict in self._fill_feed_dict(partition, reshuffle=False):
                self.sess.run([accuracy_update_op], feed_dict=feed_dict)
            accuracy = self.sess.run([accuracy_op])[0]
        self.sess.run([tf.variables_initializer(stream_vars)])
        return accuracy
