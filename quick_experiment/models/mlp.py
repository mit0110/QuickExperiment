import os
import numpy
import sys
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.ERROR)

from quick_experiment import utils
from quick_experiment.model import BaseModel


class MLPModel(BaseModel):
    """A MultiLayer Perceptron model.

    Args:
        dataset (:obj: BaseDataset): An instance of BaseDataset (or
            subclass). The dataset MUST have a partition called validation.
        hidden_layers (:obj: iterable, optional): An iterable with the size of
            the hidden layers of the network.
        batch_size (int): The maximum size of elements to input into the model.
            It will also be used to generate batches from the dataset.
        logs_dirname (string): Name of directory to save internal information
            for tensorboard visualization. If None, no records will be saved.
        log_values (int): Number of steps to wait before logging the progress of
            the training in console. If 0, no logs will be generated.
        **kwargs: Additional arguments.
    """

    def __init__(self, dataset, name=None, hidden_layers=[], batch_size=None,
                 logs_dirname='.', log_values=True,
                 learning_rate=0.001, **kwargs):
        super(MLPModel, self).__init__(dataset, **kwargs)
        self.hidden_layers_sizes = hidden_layers

        # Variable names to save the model.
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.logs_dirname = None
        if logs_dirname is not None:
            if name is None:
                self.logs_dirname = logs_dirname
            else:
                self.logs_dirname = os.path.join(logs_dirname, name)
            utils.safe_mkdir(self.logs_dirname)
        self.log_values = log_values
        self.validation_performance = []
        self.training_performance = []

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
        if self.logs_dirname is not None:
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
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
        return tf.argmax(logits, -1, name='batch_predictions')

    def build_all(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self._build_inputs()
            # Build a Graph that computes predictions from the inference model.
            logits = self._build_layers()
            # Add to the Graph the Ops for loss calculation.
            self.loss_op = self._build_loss(logits)
            # Add to the Graph the Ops that calculate and apply gradients.
            self.train_op = self._build_train_operation(self.loss_op)
            self.predictions = self._build_predictions(logits)
            self.evaluation_op = self._build_evaluation(self.predictions)

            if self.logs_dirname is not None:
                # Summarize metrics for TensorBoard
                self.summary_op = tf.summary.merge_all()

            # Create a saver for writing training checkpoints.
            self.saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            init = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))

            if self.logs_dirname is not None:
                # Instantiate a SummaryWriter to output summaries and the Graph.
                self.summary_writer = tf.summary.FileWriter(self.logs_dirname,
                                                            self.sess.graph)
            self.sess.run([init, init_local])

    def fit(self, partition_name='train', training_epochs=100,
            close_session=False):
        print('Fitting {}'.format(training_epochs))
        if self.graph is None:
            self.build_all()

        with self.graph.as_default():

            # Run the training loop
            for epoch in range(training_epochs):
                start_time = time.time()
                loss_value = self.run_train_op(epoch, self.loss_op,
                                               partition_name, self.train_op)
                end_time = time.time()

                if (epoch != 0 and self.log_values is not 0
                        and epoch % self.log_values is 0):
                    print('Classifier loss at step {} ({:.2f}s): {}'.format(
                        epoch, end_time - start_time, loss_value
                    ))
                    performance = self.evaluate()
                    print('Validation performance {}'.format(performance))
                    self.validation_performance.append((epoch, performance))
                    performance = self.evaluate('train')
                    print('Training performance {}'.format(performance))
                    self.training_performance.append((epoch, performance))

        if self.logs_dirname is not None:
            self.saver.save(
                self.sess, os.path.join(self.logs_dirname, "model.ckpt"), 0)

        if close_session:
            self.sess.close()

    def run_train_op(self, epoch, loss, partition_name, train_op):
        feed_dict = self._fill_feed_dict(partition_name)
        # Run one step of the model.  The return values are the
        # activations from the train_op (which is discarded) and
        # the loss Op.
        _, loss_value = self.sess.run([train_op, loss],
                                      feed_dict=feed_dict)
        if self.logs_dirname is not None and epoch % 10 is 0:
            self.write_summary(epoch, feed_dict)
        return loss_value

    def write_summary(self, epoch, feed_dict):
        summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, epoch)
        self.summary_writer.flush()

    def evaluate(self, partition='validation'):
        true_count = 0
        self.dataset.reset_batch(partition)
        batch = self.dataset.next_batch(self.batch_size, partition,
                                        reshuffle=False)
        while batch is not None:
            instance_batch, labels_batch = batch
            feed_dict = {
                self.instances_placeholder: instance_batch,
                self.labels_placeholder: labels_batch
            }
            true_count += self.sess.run(self.evaluation_op, feed_dict=feed_dict)
            batch = self.dataset.next_batch(self.batch_size, partition,
                                            reshuffle=False)
        return true_count / float(self.dataset.num_examples(partition))

    def predict(self, partition_name):
        predictions = []
        true = []
        self.dataset.reset_batch(partition_name)
        with self.graph.as_default():
            batch = self.dataset.next_batch(self.batch_size, 'validation',
                                            reshuffle=False)
            while batch is not None:
                instance_batch, labels_batch = batch
                feed_dict = {
                    self.instances_placeholder: instance_batch,
                    self.labels_placeholder: labels_batch
                }
                predictions.extend(self.sess.run(self.predictions,
                                                 feed_dict=feed_dict))
                true.append(labels_batch)
                batch = self.dataset.next_batch(self.batch_size, 'validation',
                                                reshuffle=False)
        return numpy.concatenate(true), numpy.array(predictions)

    def load(self, checkpoint_filename):
        """Load the model from disk"""
        if not self.graph:
            self.build_all()
        self.saver.restore(self.sess, checkpoint_filename)

    def count_trainable_parameters(self):
        with self.graph.as_default():
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print(variable.name)
                print(shape)
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print(total_parameters)
