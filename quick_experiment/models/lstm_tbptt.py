import numpy
import tensorflow as tf

from quick_experiment.models.lstm import LSTMModel


class TruncLSTMModel(LSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts a single output for a sequence. It uses truncated
    back-propagation through time. With this, it can receive sequences of
    arbitrary length.

    Args:
        dataset (:obj: SequenceDataset): An instance of SequenceDataset (or
            subclass). The dataset MUST have a partition called validation.
        hidden_layer_size (int): The size of the hidden layer of the network.
        batch_size (int): The maximum size of elements to input into the model.
            It will also be used to generate batches from the dataset.
        logs_dirname (string): Name of directory to save internal information
            for tensorboard visualization. If None, no records will be saved
        log_values (int): Number of steps to wait before logging the progress
            of the training in console. If 0, no logs will be generated.
        max_num_steps (int): the maximum number of steps to use during the
            Back Propagation Through Time optimization. The gradients are
            going to be clipped at max_num_steps.
        **kwargs: Additional arguments.
    """

    def __init__(self, dataset, use_prev_state=True, **kwargs):
        super(TruncLSTMModel, self).__init__(
            dataset, **kwargs)
        self.current_batch_size = None
        self.batch_lengths = None
        self.use_prev_state = use_prev_state

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.float32,
            (None, None, self.dataset.feature_vector_size),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (None, ),
            name='labels_placeholder')

    def reshape_output(self, outputs, lengths):
        """Transforms the network hidden layer output into the input for the
        softmax layer.

        Args:
            outputs (Tensor): shape [batch_size, max_time, cell.output_size].
            lengths (Tensor): shape [batch_size]

        Returns:
            A tensor with shape [batch_size, cell.output_size].
        """
        batch_size = tf.shape(outputs)[0]
        max_length = tf.shape(outputs)[1]
        out_size = int(outputs.get_shape()[2])
        index = (tf.range(0, batch_size, dtype=tf.int32) * max_length +
                 (tf.minimum(lengths, max_length) - 1))
        # index has shape (batch_size,)
        flat = tf.reshape(outputs, [-1, out_size])
        # flat has shape [batch_size * max_time, cell.output_size]
        relevant = tf.gather(flat, index)
        return relevant

    def _pad_batch(self, input_tensor):
        self.current_batch_size = tf.shape(input_tensor)[0]
        new_instances = tf.subtract(self.batch_size, tf.shape(input_tensor)[0])
        # Pad lenghts
        self.batch_lengths = tf.pad(self.lengths_placeholder,
                                    paddings=[[tf.constant(0), new_instances]],
                                    mode='CONSTANT')
        # Pad instances
        paddings = [[tf.constant(0), new_instances],
                     tf.constant([0, 0]), tf.constant([0, 0])]
        input_tensor = tf.pad(input_tensor, paddings=paddings, mode='CONSTANT')
        # Ensure the correct shape. This is only to avoid an error with the
        # dynamic_rnn, which needs to know the size of the batch.
        return tf.reshape(
            input_tensor, shape=(self.batch_size, self.max_num_steps,
                                 self.dataset.feature_vector_size))

    def _build_input_layers(self):
        # Since the model needs a fixed batch size, we extend the
        # instances_placeholder with zeros. For training, we are sure all
        # batches will be the same size as the reshuffle option has to be true.
        input_tensor = self._pad_batch(self.instances_placeholder)

        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')
        if self.dropout_ratio != 0:
            return tf.layers.dropout(
                inputs=tf.cast(input_tensor, tf.float32),
                rate=self.dropout_placeholder)
        return input_tensor

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        input = self._build_input_layers()
        output = self._build_recurrent_layer(input)

        if self.dropout_ratio != 0:
            output = tf.layers.dropout(
                inputs=output, rate=self.dropout_placeholder)

        # Reshape again to real batch size
        output = output[:self.current_batch_size, :]

        # The last layer is for the classifier
        with tf.name_scope('logits_layer') as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=output, num_outputs=self.dataset.classes_num(),
                activation_fn=None,  # we keep a linear activation
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                trainable=True, scope=scope
            )
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

    def _build_rnn_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self.hidden_layer_size, forget_bias=1.0)

    def _build_recurrent_layer(self, input_op):
        # The recurrent layer
        lstm_cell = self._build_rnn_cell()
        with tf.name_scope('recurrent_layer') as scope:
            # Get the initial state. States will be a LSTMStateTuples.
            state_variable = self._build_state_variables(lstm_cell)
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, new_state = tf.nn.dynamic_rnn(
                lstm_cell, inputs=tf.cast(input_op, tf.float32),
                sequence_length=self.batch_lengths, scope=scope,
                initial_state=state_variable)
            last_output = self.reshape_output(outputs, self.batch_lengths)
            # We take only the last predicted output
            # Define the state operations. This wont execute now.
            self.last_state_op = self._get_state_update_op(state_variable,
                                                           new_state)
            self.reset_state_op = self._get_state_update_op(
                state_variable,
                lstm_cell.zero_state(self.batch_size, tf.float32))
        return last_output

    def log_gradients(self, gradients):
        for gradient, variable in gradients:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            # tf.summary.scalar(variable.name, tf.reduce_sum(variable))
            tf.summary.histogram(variable.name + "/gradients", grad_values)

    def _build_train_operation(self, loss):
        if self.logs_dirname is not None:
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer()
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        gradients = optimizer.compute_gradients(loss)
        if self.log_gradients and self.logs_dirname is not None:
            self._log_gradients_op(gradients)
        return optimizer.apply_gradients(gradients, global_step=global_step)

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
            step_size=self.max_num_steps, reshuffle=reshuffle)
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
        step_lengths = numpy.clip(
            numpy.where(lengths >= start_index, lengths - start_index, 0),
            a_max=self.max_num_steps, a_min=0)
        feed_dict = {
            self.instances_placeholder: step_instances,
            self.lengths_placeholder: step_lengths,
            self.labels_placeholder: labels
        }
        return feed_dict

    def run_train_op(self, epoch, loss, partition_name, train_op):
        # Reset the neural_network_value
        self.sess.run(self.reset_state_op)
        # We need to run the train op cutting the sequence in chunks of
        # max_steps size
        loss_value = []
        feed_dict = None
        for feed_dict in self._fill_feed_dict(partition_name):
            feed_dict[self.dropout_placeholder] = self.dropout_ratio
            if self.use_prev_state:
                result = self.sess.run(
                    [train_op, loss, self.last_state_op], feed_dict=feed_dict)
            else:
                self.sess.run(self.reset_state_op)
                result = self.sess.run(
                    [train_op, loss], feed_dict=feed_dict)
            loss_value.append(result[1])
        if self.logs_dirname is not None and epoch % 10 is 0 and feed_dict:
            self.write_summary(epoch, feed_dict)
        return numpy.mean(loss_value)

    def predict(self, partition_name, limit=-1):
        predictions = []
        true = []
        old_start = self.dataset.reset_batch(partition_name)
        with self.graph.as_default():
            while (self.dataset.has_next_batch(self.batch_size, partition_name)
                   and (limit <= 0 or len(predictions) < limit)):
                batch_prediction = numpy.zeros(shape=self.batch_size) - 1
                batch_true = None
                # This for loop will execute at least once
                # We need to feed all steps to the network before getting the
                # last prediction.
                for feed_dict in self._fill_feed_dict(partition_name,
                                                      reshuffle=False):
                    # Labels will be always the same for all steps in batch
                    batch_true = feed_dict[self.labels_placeholder]
                    step_prediction = self.sess.run(self.predictions,
                                                    feed_dict=feed_dict)
                    # Get the last prediction
                    for index, length in enumerate(
                            feed_dict[self.lengths_placeholder]):
                        if length > 0:
                            batch_prediction[index] = step_prediction[index]
                    assert (batch_prediction.shape[0] == self.batch_size or
                            batch_prediction.shape[0] == batch_true.shape[0])
                assert batch_true is not None
                predictions.extend(batch_prediction[:batch_true.shape[0]])
                true.extend(batch_true)
        self.dataset.reset_batch(partition_name, old_start)
        return numpy.array(true), numpy.array(predictions)

    def _build_evaluation(self, predictions):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            predictions: Predictions tensor, int - [current_batch_size,
                max_num_steps].
        Returns:
            A scalar int32 tensor with the number of examples (out of
            batch_size) that were predicted correctly.
        """
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_performance'):
            # We use the mask to ignore predictions outside the sequence length.
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                predictions, self.labels_placeholder)

        return accuracy, accuracy_update

    def evaluate(self, partition='validation'):
        with self.graph.as_default():
            # Reset the accuracy variables
            stream_vars = [i for i in tf.local_variables()
                           if i.name.split('/')[0] == 'evaluation_performance']
            self.sess.run([tf.variables_initializer(stream_vars)])
            metric_op, metric_update_op = self.evaluation_op
            old_start = self.dataset.reset_batch(partition)
            metric = None
            while self.dataset.has_next_batch(self.batch_size, partition):
                for feed_dict in self._fill_feed_dict(partition,
                                                      reshuffle=False):
                    self.sess.run([metric_update_op], feed_dict=feed_dict)
                metric = self.sess.run([metric_op])[0]
            self.dataset.reset_batch(partition, old_start)
        return metric
