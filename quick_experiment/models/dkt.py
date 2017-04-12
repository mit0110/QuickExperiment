import numpy
import tensorflow as tf

from lstm import SeqPredictionModel


class DKTModel(SeqPredictionModel):

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.float32,
            (None, self.max_num_steps, self.dataset.feature_vector_size),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        # Each label is an element number and the output of the interaction.
        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (None, self.max_num_steps, 2),
            name='labels_placeholder')

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]"""
        logits_shape = tf.shape(logits)
        # Calculate the cross entropy for the prediction of each step.
        # We want to compare the true output against the predicted probability
        # of the labeled problem
        logits = tf.log(tf.nn.softmax(logits))
        # Separate the  index for each problem
        indices = tf.reshape(tf.slice(self.labels_placeholder, [0, 0, 0],
                                      [-1, logits_shape[1], 1]),
                             [self.batch_size, self.max_num_steps])
        # Filter out only the logits corresponding to the problem.
        label_encodings = tf.one_hot(indices=indices, depth=logits_shape[-1],
                                     dtype=logits.dtype)
        cross_entropy = tf.multiply(logits, label_encodings)
        # cross_entropy has shape [batch_size, max_num_steps, feature_size]
        # Now we sum over all classes (in this case, it's equivalent of taking
        # the max since only one element of y_true is active per example
        cross_entropy = tf.reduce_sum(cross_entropy, axis=2)
        # cross_entropy has shape [batch_size, max_num_steps]

        true_outpus = tf.reshape(tf.slice(self.labels_placeholder, [0, 0, 1],
                                          [-1, logits_shape[1], 1]),
                                 [self.batch_size, self.max_num_steps])
        cross_entropy = -tf.multiply(cross_entropy,
                                     tf.cast(true_outpus, cross_entropy.dtype))

        cross_entropy = tf.multiply(cross_entropy, tf.sequence_mask(
            self.lengths_placeholder, logits_shape[1], dtype=logits.dtype))

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
                    feed_dict[self.labels_placeholder][index, :length, 1]
                    for index, length in enumerate(lengths)])
        return numpy.array(predictions), numpy.array(true)

    def _build_predictions(self, logits):
        """Return a tensor with the predicted class for each instance.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                NUM_CLASSES].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps]. The prediction for each element is the probability
            of doing the true next excercise correctly
        """
        logits = tf.nn.softmax(logits)
        indices = tf.reshape(tf.slice(self.labels_placeholder, [0, 0, 0],
                                      [-1, tf.shape(logits)[1], 1]),
                             [self.batch_size, self.max_num_steps])
        # Filter out only the logits corresponding to the problem.
        label_encodings = tf.one_hot(
            indices=indices, depth=tf.shape(logits)[-1], dtype=logits.dtype)
        filtered_logits = tf.multiply(logits, label_encodings)
        # filtered_logits has shape [batch_size, max_num_steps, feature_size]
        # but has only one active record per element of each sequence.
        return tf.reduce_max(filtered_logits, -1, name='batch_predictions')

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
        true_outpus = tf.reshape(
            tf.slice(self.labels_placeholder, [0, 0, 1],
                     [-1, self.max_num_steps, 1]),
            [self.batch_size, self.max_num_steps])
        with tf.name_scope('evaluation_auc'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            auc_value, auc_update = tf.contrib.metrics.streaming_auc(
                predictions, true_outpus, weights=mask, num_thresholds=50)

        return auc_value, auc_update

    def evaluate_validation(self, correct_predictions):
        # Reset the auc variables
        stream_vars = [i for i in tf.local_variables()
                       if i.name.split('/')[0] == 'evaluation_auc']
        auc_value_op, auc_update_op = correct_predictions
        for feed_dict in self._fill_feed_dict_traversing('validation'):
            self.sess.run([auc_update_op], feed_dict=feed_dict)
        auc_value = self.sess.run(auc_value_op)
        self.sess.run([tf.variables_initializer(stream_vars)])
        return auc_value

