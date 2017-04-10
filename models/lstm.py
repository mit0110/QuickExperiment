import logging

import numpy
import tensorflow as tf
import tflearn

from models.mlp import MLPModel

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO)


class LSTMModel(MLPModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts a single output for a sequence.

    Args:
        dataset (:obj: SequenceDataset): An instance of SequenceDataset (or
            subclass). The dataset MUST have a partition called validation.
        hidden_layer_size (int): The size of the hidden layer of the network.
        batch_size (int): The maximum size of elements to input into the model.
            It will also be used to generate batches from the dataset.
        training_epochs (int): Number of training iterations
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
                 training_epochs=1000, logs_dirname='.', log_values=True,
                 max_num_steps=30, **kwargs):
        super(LSTMModel, self).__init__(
            dataset, batch_size=batch_size, training_epochs=training_epochs,
            logs_dirname=logs_dirname, name=name, log_values=log_values,
            **kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.max_num_steps = max_num_steps

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

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        output = self._build_recurrent_layer()
        # The last layer is for the classifier
        with tf.name_scope('logits_layer') as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=output, num_outputs=self.dataset.classes_num(),
                activation_fn=None,  # we keep a linear activation
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                reuse=True, trainable=True, scope=scope
            )
            if self.logs_dirname is not None:
                tf.summary.histogram('logits', logits)
        return logits

    def _build_recurrent_layer(self):
        # The recurrent layer
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.hidden_layer_size, forget_bias=1.0)
        with tf.name_scope('recurrent_layer') as scope:
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, state = tf.nn.dynamic_rnn(
                lstm_cell, inputs=self.instances_placeholder,
                sequence_length=self.lengths_placeholder, scope=scope,
                initial_state=lstm_cell.zero_state(
                    tf.shape(self.instances_placeholder)[0], tf.float32))
            last_output = self.reshape_output(outputs, self.lengths_placeholder)
            # We take only the last predicted output
            if self.logs_dirname is not None:
                tf.summary.histogram('hidden_state', state)
                tf.summary.histogram('outputs', outputs)
        return last_output

    def log_gradients(self, gradients):
        if self.logs_dirname is None:
            return
        # Add histograms for variables, gradients and gradient norms.
        for gradient, variable in gradients:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + "/gradients",
                                 grad_values)

    def _build_train_operation(self, loss):
        print loss
        if self.logs_dirname is not None:
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        gradients = optimizer.compute_gradients(loss)
        self.log_gradients(gradients)
        return optimizer.apply_gradients(gradients, global_step=global_step)

    def _fill_feed_dict(self, partition_name='train'):
        """Fills the feed_dict for training the given step.

        Args:
            partition_name (string): The name of the partition to get the batch
                from.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        instances, labels, lengths = self.dataset.next_batch(
            self.batch_size, partition_name, pad_sequences=True,
            max_sequence_length=self.max_num_steps)
        result = {
            self.instances_placeholder: instances.astype(numpy.float32),
            self.lengths_placeholder: lengths
        }
        if hasattr(self, 'labels_placeholder') and labels is not None:
            result[self.labels_placeholder] = labels
        return result

    def _fill_feed_dict_traversing(self, partition_name='train'):
        """Fills the feed_dict for training the given step.

        Args:
            partition_name (string): The name of the partition to get the batch
                from.

        Yields:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        for instances, labels, lengths in self.dataset.traverse_dataset(
                self.batch_size, partition_name,
                max_sequence_length=self.max_num_steps):
            feed_dict = {
                self.instances_placeholder: instances,
                self.labels_placeholder: labels,
                self.lengths_placeholder: lengths
            }
            yield feed_dict

    def predict(self, partition_name):
        predictions = []
        true = []
        with self.graph.as_default():
            for feed_dict in self._fill_feed_dict_traversing(partition_name):
                predictions.extend(self.sess.run(self.predictions,
                                                 feed_dict=feed_dict))
                true.append(feed_dict[self.labels_placeholder])
        return numpy.array(predictions), numpy.concatenate(true)

    def evaluate_validation(self, correct_predictions):
        true_count = 0
        for feed_dict in self._fill_feed_dict_traversing('validation'):
            true_count += self.sess.run(correct_predictions,
                                        feed_dict=feed_dict)
        return true_count / float(self.dataset.num_examples('validation'))


class SeqPredictionModel(LSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The dataset
    must not have labels. The labels will be the next element of the sequence
    or the end of sequence symbol.
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
            self.dataset.labels_type, (None, self.max_num_steps),
            name='labels_placeholder')

    def reshape_output(self, outputs, lengths):
        """Transforms the network hidden layer output into the input for the
        softmax layer.

        Args:
            outputs (Tensor): shape [batch_size, max_time, cell.output_size].

        Returns:
            A tensor with shape [batch_size, cell.output_size].
        """
        return outputs

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        # The recurrent layer
        output = self._build_recurrent_layer()
        # outputs is a Tensor shaped
        # [batch_size, max_num_steps, hidden_size].

        # The last layer is for the classifier
        logits = tflearn.layers.core.time_distributed(
            output, tflearn.layers.core.fully_connected,
            args=[self.dataset.classes_num()],
            scope='softmax_layer')  # [batch_size, max_num_steps, classes_num]
        # logits is now a tensor [batch_size, max_num_steps, classes_num]

        return logits

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]"""
        logits_shape = logits.get_shape()
        assert self.labels_placeholder.get_shape()[1] == logits_shape[1]
        # Calculate the cross entropy for the prediction of each step.
        # The (binary) cross entropy is defined as
        # -\sum_{x \in {a, b}} p(x) log(q)
        # We want to compare the true label against the predicted probability
        # of that label
        logits = tf.log(tf.nn.softmax(logits))
        label_encodings = tf.one_hot(indices=self.labels_placeholder,
                                     depth=logits_shape[-1],
                                     dtype=logits.dtype)

        cross_entropy = -tf.multiply(logits, label_encodings)
        # cross_entropy has shape [batch_size, max_num_steps, feature_size] but
        # has only one active record per element of each sequence

        # Now we sum over all classes (in this case, it's equivalent of taking
        # the max since only one element of y_true is active per example
        cross_entropy = tf.reduce_sum(cross_entropy, axis=2)
        cross_entropy = tf.multiply(cross_entropy, tf.sequence_mask(
            self.lengths_placeholder, logits_shape[1], dtype=logits.dtype))
        # cross_entropy has shape [batch_size, max_num_steps]

        # Remove the elements that are not part of the sequence.
        # We take the average cross entropy of each sequence, taking into
        # consideration the lenght of the sequence.
        cross_entropy = (tf.reduce_sum(cross_entropy, axis=1) /
                         tf.to_float(self.lengths_placeholder))
        # cross_entropy now has shape [batch_size, ]

        # Finally, we take the average over all examples in batch.
        return tf.reduce_mean(cross_entropy)

    def predict(self, partition_name):
        predictions = []
        true = []
        with self.graph.as_default():
            for feed_dict in self._fill_feed_dict_traversing(partition_name):
                lengths = feed_dict[self.lengths_placeholder]
                batch_prediction = self.sess.run(self.predictions,
                                                 feed_dict=feed_dict)
                # batch prediction has shape [batch_size, max_num_steps]
                # now we need to return only the predictions in the sequence
                predictions.extend([batch_prediction[index, :length]
                                    for index, length in enumerate(lengths)])
                true.extend([
                    feed_dict[self.labels_placeholder][index, :length]
                    for index, length in enumerate(lengths)])

        return numpy.array(predictions), numpy.array(true)

    def _build_predictions(self, logits):
        """Return a tensor with the predicted class for each instance.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                NUM_CLASSES (feature_vector_size + 1)].

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
        # self.labels is a binary vector. Then correct has the values of the
        # predictions in the correct labels only
        predictions = self._build_predictions(logits)
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_accuracy'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                predictions, self.labels_placeholder, weights=mask)

        return accuracy, accuracy_update

    def evaluate_validation(self, correct_predictions):
        # Reset the accuracy variables
        stream_vars = [i for i in tf.local_variables()
                       if i.name.split('/')[0] == 'evaluation_accuracy']
        accuracy_op, accuracy_update_op = correct_predictions
        for feed_dict in self._fill_feed_dict_traversing('validation'):
            self.sess.run([accuracy_update_op], feed_dict=feed_dict)
        accuracy = self.sess.run([accuracy_op])
        self.sess.run([tf.variables_initializer(stream_vars)])
        return accuracy[0]

