import logging
import os
import numpy
import tensorflow as tf

import utils
from model import BaseModel

logging.basicConfig(level=logging.INFO)


class LSTMModel(BaseModel):
    """A Recurrent Neural Network model with LSTM cells.

    Args:
        dataset (:obj: BaseDataset): An instance of BaseDataset (or
            subclass). The dataset MUST have a partition called validation.
        hidden_layer_size (int): The size of the hidden layer of the network.
        batch_size (int): The maximum size of elements to input into the model.
            It will also be used to generate batches from the dataset.
        training_epochs (int): Number of training iterations
        logs_dirname (string): Name of directory to save internal information
            for tensorboard visualization.
        log_valus (bool): If True, log the progress of the training in console.
        max_num_steps (int): the maximum number of steps to use during the
            Back Propagation Through Time optimization. The gradients are
            going to be clipped at max_num_steps.
        **kwargs: Additional arguments.
    """

    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 training_epochs=1000, logs_dirname='.', log_values=True,
                 max_num_steps=30, **kwargs):
        super(LSTMModel, self).__init__(dataset, **kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.max_num_steps = max_num_steps

        # Variable names to save the model.
        self.learning_rate = 0.01
        self.batch_size = batch_size
        self.training_epochs = training_epochs

        if name is None:
            self.logs_dirname = logs_dirname
        else:
            self.logs_dirname = os.path.join(logs_dirname, name)
        utils.safe_mkdir(self.logs_dirname)

        self.log_values = log_values

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.sequences_placeholder = tf.placeholder(
            self.dataset.instances_type,
            (None, self.max_num_steps, self.dataset.feature_vector_size),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (None, ),
            name='labels_placeholder')

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        # The recurrent layer
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size)

        with tf.name_scope('recurrent_layer') as scope:
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, state = tf.nn.dynamic_rnn(
                lstm_cell, inputs=self.sequences_placeholder,
                sequence_length=self.sequences_placeholder, scope=scope)

        # The last layer is for the classifier
        with tf.name_scope('softmax_layer') as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=layers[-1], num_outputs=self.dataset.classes_num(),
                activation_fn=tf.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=tf.contrib.layers.l2_regularizer,
                reuse=True, trainable=True, scope=scope
            )
            tf.summary.histogram('logits', logits)

        return logits

    def _build_loss(self, logits):
        """Calculates the loss from the logits and the labels.

        Args:
            logits: Logits tensor, float - [self.batch_size, NUM_CLASSES].
            labels_placeholder: Labels tensor, int32 - [batch_size]

        Returns:

        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder, name='cross_entropy')
        return tf.reduce_mean(cross_entropy, name='cross_entropy_mean_loss')

    def _build_train_operation(self, loss):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        return optimizer.minimize(loss, global_step=global_step)


    def _fill_feed_dict(self, partition_name='train'):
        """Fills the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
          <placeholder>: <tensor of values to be passed for placeholder>,
          ....
        }

        Args:
            partition_name (string): The name of the partition to get the batch
                from.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        instances, labels = self.dataset.next_batch(
            self.batch_size, partition_name)
        return {
            self.instances_placeholder: instances,
            self.labels_placeholder: labels,
        }

    def _build_evaluation(self, logits):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
                range [0, NUM_CLASSES).

        Returns:
            A scalar int32 tensor with the number of examples (out of batch_size)
            that were predicted correctly.
        """
        correct = tf.nn.in_top_k(logits, self.labels_placeholder, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def _build_predictions(self, logits):
        """Return a tensor with the predicted class for each instance.

        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].

        Returns:
            A int32 tensor with the predictions.
        """
        # Variable to store the predictions
        return tf.argmax(tf.nn.softmax(logits), 1, name='batch_predictions')

    def fit(self, partition_name='train', close_session=False):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self._build_inputs()
            # Build a Graph that computes predictions from the inference model.
            logits = self._build_layers()
            # Add to the Graph the Ops for loss calculation.
            loss = self._build_loss(logits)
            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self._build_train_operation(loss)
            correct_predictions = self._build_evaluation(logits)
            self.predictions = self._build_predictions(logits)

            # Summarize metrics for TensorBoard
            summary = tf.summary.merge_all()

            # Create a saver for writing training checkpoints.
            self.saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(self.logs_dirname,
                                                   self.sess.graph)
            self.sess.run(init)

            # Run the training loop
            for epoch in range(self.training_epochs):
                feed_dict = self._fill_feed_dict(partition_name)

                # Run one step of the model.  The return values are the
                # activations from the train_op (which is discarded) and
                # the loss Op.
                _, loss_value = self.sess.run([train_op, loss],
                                         feed_dict=feed_dict)
                if epoch % 10 is 0:
                    summary_str = self.sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, epoch)
                    summary_writer.flush()

                if self.log_values and epoch % 100 is 0:
                    logging.info('Classifier loss at step {}: {}'.format(
                        epoch, loss_value
                    ))
                    accuracy = self.evaluate_validation(correct_predictions)
                    tf.summary.scalar('accuracy', accuracy)
                    logging.info('Validation accuracy {}'.format(accuracy))

        if close_session:
            self.sess.close()

    def evaluate_validation(self, correct_predictions):
        true_count = 0
        for instance_batch, labels_batch in self.dataset.traverse_dataset(
                self.batch_size, 'validation'):
            feed_dict = {
                self.instances_placeholder: instance_batch,
                self.labels_placeholder: labels_batch
            }
            true_count += self.sess.run(correct_predictions,
                                        feed_dict=feed_dict)
        return true_count / float(self.dataset.num_examples('validation'))

    def predict(self, partition_name):
        predictions = []
        with self.graph.as_default():
            for instance_batch, labels_batch in self.dataset.traverse_dataset(
                    self.batch_size, partition_name):
                feed_dict = {
                    self.instances_placeholder: instance_batch,
                    self.labels_placeholder: labels_batch
                }
                predictions.extend(self.sess.run(self.predictions,
                                                 feed_dict=feed_dict))
        return numpy.array(predictions)