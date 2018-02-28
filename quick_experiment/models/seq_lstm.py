"""Model for sequence prediction with a LSTM RNN."""
import numpy
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from quick_experiment.models.lstm_tbptt import TruncLSTMModel
from quick_experiment.models.bi_lstm import BiLSTMModel


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


class SeqLSTMModel(TruncLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence.
    """

    def __init__(self, dataset, **kwargs):
        super(SeqLSTMModel, self).__init__(dataset, **kwargs)
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

        # Reshape again to real batch size
        output = output[:self.current_batch_size, :, :]
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

    def _build_rnn_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size)

    def _build_recurrent_layer(self, input_op):
        # The recurrent layer
        rnn_cell = self._build_rnn_cell()
        with tf.name_scope('recurrent_layer') as scope:
            # Get the initial state. States will be a LSTMStateTuples.
            state_variable = self._build_state_variables(rnn_cell)
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, new_state = tf.nn.dynamic_rnn(
                rnn_cell, inputs=input_op,
                sequence_length=self.batch_lengths, scope=scope,
                initial_state=state_variable)
            # Define the state operations. This wont execute now.
            self.last_state_op = self._get_state_update_op(state_variable,
                                                           new_state)
            self.reset_state_op = self._get_state_update_op(
                state_variable,
                rnn_cell.zero_state(self.batch_size, tf.float32))
        return tf.concat(outputs, axis=1) 

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]
        """
        mask = tf.sequence_mask(self.batch_lengths, self.max_num_steps)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder)
        loss = tf.div(
            tf.reduce_sum(tf.boolean_mask(loss, mask)),
            tf.cast(tf.reduce_sum(self.batch_lengths), loss.dtype))
        return loss

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

    def _build_evaluation(self, predictions):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            predictions: Predictions tensor, int - [current_batch_size,
                max_num_steps].
        Returns:
            A scalar int32 tensor with the number of examples (out of
            batch_size) that were predicted correctly.
        """
        # predictions has shape [current_batch_size, max_num_steps]
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
        old_start = self.dataset.reset_batch(partition_name)
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
        self.dataset.reset_batch(partition_name, old_start)
        return numpy.array(true), numpy.array(predictions)


class SeqBiLSTMModel(SeqLSTMModel):
    """A Recurrent Neural Network model with LSTM bidirectional cells.

    Predicts the probability of the next element on the sequence."""

    def _build_rnn_cell(self):
        return (tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size,
                                             forget_bias=1.0),
                tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size,
                                             forget_bias=1.0))

    def _build_state_variables(self, cell_fw, cell_bw):
        # Get the initial state and make a variable out of it
        # to enable updating its value.
        state_c_fw, state_h_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        state_c_bw, state_h_bw = cell_bw.zero_state(self.batch_size, tf.float32)
        return (
            tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c_fw, trainable=False),
                tf.Variable(state_h_fw, trainable=False)),
            tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c_bw, trainable=False),
                tf.Variable(state_h_bw, trainable=False)),
        )

    @staticmethod
    def _get_state_update_op(state_variables, new_state_fw, new_state_bw):
        # Add an operation to update the train states with the last state
        # Assign the new state to the state variables on this layer
        state_fw, state_bw = state_variables
        return (state_fw[0].assign(new_state_fw[0]),
                state_fw[1].assign(new_state_fw[1]),
                state_bw[0].assign(new_state_bw[0]),
                state_bw[1].assign(new_state_bw[1])
                )

    def _build_recurrent_layer(self, input_op):
        # The recurrent layer
        lstm_cell_fw, lstm_cell_bw = self._build_rnn_cell()
        with tf.name_scope('recurrent_layer') as scope:
            state_variables = self._build_state_variables(lstm_cell_fw,
                                                         lstm_cell_bw)
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            (output_fw, output_bw), new_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, dtype=tf.float32,
                inputs=tf.cast(input_op, tf.float32,),
                sequence_length=self.batch_lengths, scope=scope,
                initial_state_fw=state_variables[0],
                initial_state_bw=state_variables[1])
            # We take only the last predicted output. Each output has shape
            # [batch_size, cell.output_size], and when we concatenate them
            # the result has shape [batch_size, 2*cell.output_size]
            self.last_state_op = self._get_state_update_op(state_variables,
                                                           *new_states)
            self.reset_state_op = self._get_state_update_op(
                state_variables,
                lstm_cell_fw.zero_state(self.batch_size, tf.float32),
                lstm_cell_bw.zero_state(self.batch_size, tf.float32))
        return tf.concat([output_fw, output_bw], axis=1)

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        input = self._build_input_layers()
        output = self._build_recurrent_layer(input)
        # outputs is a Tensor shaped
        # [batch_size, max_num_steps, hidden_size].

        # Reshape again to real batch size
        output = output[:self.current_batch_size, :, :]
        # The last layer is for the classifier. BIDIRECTIONAL state is
        # double size
        output = tf.reshape(output, [-1, self.hidden_layer_size*2])
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

