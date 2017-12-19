import numpy
import tensorflow as tf

from quick_experiment.models.mlp import MLPModel


class LSTMModel(MLPModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts a single output for a sequence.

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

    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 logs_dirname='.', log_values=True, dropout_ratio=0.3,
                 max_num_steps=30, **kwargs):
        super(LSTMModel, self).__init__(
            dataset, batch_size=batch_size, logs_dirname=logs_dirname,
            name=name, log_values=log_values, **kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.max_num_steps = max_num_steps
        self.dropout_ratio = dropout_ratio

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

    def _build_input_layers(self):
        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')
        if self.dropout_ratio != 0:
            return tf.layers.dropout(
                inputs=tf.cast(self.instances_placeholder, tf.float32),
                rate=self.dropout_placeholder)
        return self.instances_placeholder

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        input = self._build_input_layers()
        output = self._build_recurrent_layer(input)

        if self.dropout_ratio != 0:
            output = tf.layers.dropout(
                inputs=output, rate=self.dropout_placeholder)
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

    def _build_recurrent_layer(self, input):
        # The recurrent layer
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.hidden_layer_size, forget_bias=1.0)
        with tf.name_scope('recurrent_layer') as scope:
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, state = tf.nn.dynamic_rnn(
                lstm_cell, inputs=input,
                sequence_length=self.lengths_placeholder, scope=scope,
                initial_state=lstm_cell.zero_state(
                    tf.shape(self.instances_placeholder)[0], tf.float32))
            last_output = self.reshape_output(outputs, self.lengths_placeholder)
            # We take only the last predicted output
        return last_output

    def log_gradients(self, gradients):
        if self.logs_dirname is None:
            return
        for gradient, variable in gradients:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.summary.scalar(variable.name, tf.reduce_sum(variable))
            tf.summary.scalar(variable.name + "/gradients",
                              tf.reduce_sum(grad_values))

    def _build_train_operation(self, loss):
        if self.logs_dirname is not None:
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer()
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        gradients = optimizer.compute_gradients(loss)
        # self.log_gradients(gradients)
        return optimizer.apply_gradients(gradients, global_step=global_step)

    def _fill_feed_dict(self, partition_name='train', reshuffle=True):
        """Fills the feed_dict for training the given step.

        Args:
            partition_name (string): The name of the partition to get the batch
                from.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        batch = self.dataset.next_batch(
            self.batch_size, partition_name, pad_sequences=True,
            max_sequence_length=self.max_num_steps, reshuffle=reshuffle)
        if batch is None:
            return None
        instances, labels, lengths = batch
        result = {
            self.instances_placeholder: instances.astype(numpy.float32),
            self.lengths_placeholder: lengths
        }
        if hasattr(self, 'labels_placeholder') and labels is not None:
            result[self.labels_placeholder] = labels
        return result

    def predict(self, partition_name):
        predictions = []
        true = []
        self.dataset.reset_batch()
        with self.graph.as_default():
            feed_dict = self._fill_feed_dict(partition_name, reshuffle=False)
            while feed_dict is not None:
                predictions.extend(self.sess.run(self.predictions,
                                                 feed_dict=feed_dict))
                true.append(feed_dict[self.labels_placeholder])
                feed_dict = self._fill_feed_dict(partition_name,
                                                 reshuffle=False)
                print(len(predictions))
        return numpy.concatenate(true), numpy.array(predictions)

    def evaluate(self, partition='validation'):
        """Returns the accuracy of the model over the given partition.

        Args:
            parition: String, one of the dataset partitions.
        """
        true_count = 0
        self.dataset.reset_batch()
        feed_dict = self._fill_feed_dict(partition, reshuffle=False)
        while feed_dict is not None:
            true_count += self.sess.run(self.evaluation_op, feed_dict=feed_dict)
            feed_dict = self._fill_feed_dict(partition, reshuffle=False)
        return true_count / float(self.dataset.num_examples(partition))

