import logging
import os
import numpy
import tensorflow as tf

import utils
from model import BaseModel

logging.basicConfig(level=logging.INFO)


class MLPModel(BaseModel):
    """A MultiLayer Perceptron model.

    Args:
        dataset (:obj: BaseDataset): An instance of BaseDataset (or
            subclass). The dataset MUST have a partition called validation.
        layers (:obj: iterable, optional): An iterable with the size of the
            hidden layers of the network.
        **kwargs: Additional arguments.
    """

    def __init__(self, dataset, name=None, hidden_layers=[], batch_size=None,
                 training_epochs=1000, logs_dirname='.', log_values=True,
                 **kwargs):
        super(MLPModel, self).__init__(dataset, **kwargs)
        self.hidden_layers_sizes = hidden_layers

        # Variable names to save the model.
        self.variable_names = {}
        self.learning_rate = 0.01
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        if name is None:
            self.logs_dirname = logs_dirname
        else:
            self.logs_dirname = os.path.join(logs_dirname, name)
        self.log_values = log_values

        utils.safe_mkdir(self.logs_dirname)

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        code and will be fed in the .fit() loop, below.
        """
        self.instances_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.dataset.feature_vector_size),
            name='instances_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, ),
            name='labels_placeholder')

    def _build_layers(self):
        """Builds the model up to the logits calculation"""
        layers = [self.instances_placeholder]
        for layer_index, size_current in enumerate(self.hidden_layers_sizes):
            layer_name = 'hidden_layer_{}'.format(layer_index)
            with tf.variable_scope(layer_name) as scope:
                # Create the layer
                layer = tf.contrib.layers.fully_connected(
                    inputs=layers[-1], num_outputs=size_current,
                    activation_fn=tf.sigmoid,
                    weights_initializer=tf.truncated_normal_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer,
                    biases_initializer=tf.zeros_initializer(),
                    biases_regularizer=tf.contrib.layers.l2_regularizer,
                    reuse=True, trainable=True, scope=scope
                )
                layers.append(layer)

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