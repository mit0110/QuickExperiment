"""Model for sequence prediction with a LSTM RNN."""
import numpy
import tensorflow as tf

from quick_experiment.models import seq_lstm


class EmbeddedSeqLSTMModel(seq_lstm.SeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.
    """
    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 training_epochs=1000, logs_dirname='.', log_values=True,
                 max_num_steps=30, embedding_size=200, dropout_ratio=0.3,
                 **kwargs):
        super(EmbeddedSeqLSTMModel, self).__init__(
            dataset, batch_size=batch_size, training_epochs=training_epochs,
            logs_dirname=logs_dirname, name=name, log_values=log_values,
            dropout_ratio=dropout_ratio, hidden_layer_size=hidden_layer_size,
            max_num_steps=max_num_steps, **kwargs)
        self.embedding_size = embedding_size
        self.output_size = embedding_size

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.int32, (None, self.max_num_steps),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            tf.int32, (None, self.max_num_steps),
            name='labels_placeholder')

    def _get_embedding(self, element_ids, element_only=False):
        """Returns self.element_embeddings + self.positive_embeddings if
        the element is positive, and self.element_embedding is is negative."""
        embedded_element = tf.nn.embedding_lookup(
            self.element_embeddings, tf.abs(element_ids))
        if element_only:
            return embedded_element
        embedded_outcome = tf.nn.embedding_lookup(
            self.positive_embedding,
            tf.clip_by_value(element_ids, clip_value_min=0,
                             clip_value_max=self.dataset.feature_vector_size))
        return tf.add_n([embedded_element, embedded_outcome])

    def _build_input_layers(self):
        with tf.name_scope('embedding_layer') as scope:
            self.element_embeddings = tf.concat([
                tf.zeros([1, self.embedding_size]),
                tf.Variable(tf.random_uniform([self.dataset.feature_vector_size,
                                               self.embedding_size], 0, 1.0),
                            trainable=True)], 0, name="element_embedding")
            self.positive_embedding = tf.concat([
                tf.Variable(tf.random_uniform([self.dataset.feature_vector_size,
                                               self.embedding_size], 0, 1.0),
                            trainable=True)], 0, name="positive_embedding")
            input = self._get_embedding(self.instances_placeholder)
            if self.dropout_ratio != 0:
                return tf.layers.dropout(inputs=input, rate=self.dropout_ratio)
            return input

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, embedding_size]
        """
        logits = tf.sigmoid(logits)
        labels = self._get_embedding(self.labels_placeholder)
        loss = tf.reduce_mean(
            tf.losses.mean_squared_error(predictions=logits, labels=labels,
                                         reduction=tf.losses.Reduction.NONE),
            axis=2)
        # calculate the mean per sequence.
        mask = tf.sequence_mask(self.lengths_placeholder, self.max_num_steps,
                                dtype=loss.dtype)
        loss = seq_lstm.safe_div(
            tf.reduce_sum(tf.multiply(loss, mask), axis=1),
            tf.cast(self.lengths_placeholder, loss.dtype))
        return tf.reduce_mean(loss)

    def _build_predictions(self, logits):
        """Return a tensor with the predicted probability for each instance.

        The probability is the norm of the distance between the softmax of the
        logits and the embedding of the true label.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                embedding_size].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps].
        """
        logits = tf.nn.sigmoid(logits)
        labels = self._get_embedding(self.labels_placeholder, element_only=True)
        predictions = tf.sigmoid(tf.norm(tf.subtract(labels, logits), ord=2,
                                         name='batch_predictions', axis=-1))
        return predictions

    def _get_step_predictions(self, batch_prediction, batch_true, feed_dict):
        step_prediction = self.sess.run(self.predictions, feed_dict=feed_dict)
        labels = numpy.clip(numpy.sign(feed_dict[self.labels_placeholder]),
                            0, 1)
        for index, length in enumerate(feed_dict[self.lengths_placeholder]):
            batch_prediction[index] = numpy.append(
                batch_prediction[index], step_prediction[index, :length])
            batch_true[index] = numpy.append(
                batch_true[index], labels[index, :length])

    def _build_evaluation(self, logits):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                embedding_size].
        Returns:
            A scalar int32 tensor with the number of examples (out of
            batch_size) that were predicted correctly.
        """
        predictions = self._build_predictions(logits)
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_r2'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            true_labels = tf.clip_by_value(
                tf.cast(tf.sign(self.labels_placeholder), predictions.dtype),
                clip_value_min=0, clip_value_max=1)
            r2, r2_update = tf.contrib.metrics.streaming_pearson_correlation(
                predictions, true_labels, weights=mask)

        return r2, r2_update


class EmbeddedSeqLSTMModel2(EmbeddedSeqLSTMModel):
    """Same model as before but with different loss."""

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, embedding_size]
        """
        logits = tf.sigmoid(logits)
        labels = self._get_embedding(self.labels_placeholder, element_only=True)
        label_outcomes = tf.cast(tf.sign(self.labels_placeholder), logits.dtype)
        # if the outcome is positive (the exercise was solved), logits is
        # bigger than the embedding of the next exercise.
        # if the outcome is negative, logits is smaller than the embeddegin of
        # the next exercise. 
        loss = tf.multiply(tf.reduce_sum(tf.subtract(logits, labels), axis=2),
                           label_outcomes)
        loss = tf.clip_by_value(loss, clip_value_min=0, clip_value_max=1)
        mask = tf.sequence_mask(
            self.lengths_placeholder, maxlen=self.max_num_steps)

        loss = tf.reduce_mean(tf.boolean_mask(loss, mask), name='loss')
        return loss

