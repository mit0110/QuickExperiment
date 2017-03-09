import logging
import os

import numpy
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import utils
from models.mlp import MLPModel

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
        log_values (int): Number of steps to wait before logging the progress of 
            the training in console. If 0, no logs will be generated.
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
    
    def reshape_output(self, outputs):
        """Transforms the network hidden layer output into the input for the 
        softmax layer.

        Args:
            outputs (Tensor): shape [batch_size, max_time, cell.output_size].

        Returns:
            A tensor with shape [batch_size, cell.output_size].
        """
        output_shape = tf.shape(outputs)
        last_output = tf.slice(
            outputs,
            begin=[0, output_shape[1] - 1, 0],
            size=[output_shape[0], 1, self.hidden_layer_size])
        last_output = tf.reshape(
            last_output, [output_shape[0], 1, self.hidden_layer_size])
        return tf.squeeze(last_output, [1])

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
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
            last_output = self.reshape_output(outputs)
            # We take only the last predicted output
            if self.logs_dirname is not None:
                tf.summary.histogram('hidden_state', state)
                tf.summary.histogram('outputs', outputs)
        # The last layer is for the classifier
        with tf.name_scope('softmax_layer') as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=last_output, num_outputs=self.dataset.classes_num(),
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
            tf.summary.histogram(variable.name + "gradients",
                                 grad_values)
            tf.summary.histogram(variable.name + "gradient_norm",
                                 tf.global_norm([grad_values]))

    def _build_train_operation(self, loss):
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
        return {
            self.instances_placeholder: instances.astype(numpy.float32),
            self.labels_placeholder: labels,
            self.lengths_placeholder: lengths
        }

    def evaluate_validation(self, correct_predictions):
        true_count = 0
        with self.graph.as_default():
            for instances, labels, lengths in self.dataset.traverse_dataset(
                    self.batch_size, 'validation'):
                feed_dict = {
                    self.instances_placeholder: instances,
                    self.labels_placeholder: labels,
                    self.lengths_placeholder: lengths
                }
                true_count += self.sess.run(correct_predictions,
                                            feed_dict=feed_dict)
        return true_count / float(self.dataset.num_examples('validation'))

    def predict(self, partition_name):
        predictions = []
        true = []
        with self.graph.as_default():
            for instances, labels, lengths in self.dataset.traverse_dataset(
                    self.batch_size, partition_name):
                feed_dict = {
                    self.instances_placeholder: instances,
                    self.labels_placeholder: labels,
                    self.lengths_placeholder: lengths
                }
                predictions.extend(self.sess.run(self.predictions,
                                                 feed_dict=feed_dict))
                true.append(labels)
        return numpy.array(predictions), numpy.concatenate(true)
