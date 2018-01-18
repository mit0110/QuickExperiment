import tensorflow as tf

from quick_experiment.models.lstm import LSTMModel


class BiLSTMModel(LSTMModel):
    """A Recurrent Neural Network model with LSTM bidirectional cells.

    Predicts a single output for a sequence.

    Args:
        dataset (:obj: SequenceDataset): An instance of SequenceDataset (or
            subclass). The dataset MUST have a partition called validation.
        hidden_layer_size (int): The size of the hidden layer of the network.
            It is used for forward and backward cells.
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

    def _build_rnn_cell(self):
        return (tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size,
                                             forget_bias=1.0),
                tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size,
                                             forget_bias=1.0))

    def _build_recurrent_layer(self, input_op):
        # The recurrent layer
        lstm_cell_fw, lstm_cell_bw = self._build_rnn_cell()
        with tf.name_scope('recurrent_layer') as scope:
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, dtype=tf.float32,
                inputs=tf.cast(input_op, tf.float32),
                sequence_length=self.batch_lengths, scope=scope)
            # We take only the last predicted output. Each output has shape
            # [batch_size, cell.output_size], and when we concatenate them
            # the result has shape [batch_size, 2*cell.output_size]
            last_output_fw = self.reshape_output(output_fw, self.batch_lengths)
            last_output_bw = self.reshape_output(output_bw, self.batch_lengths)
        return tf.concat([last_output_fw, last_output_bw], axis=1)
