from quick_experiment import dataset
import numpy
import random
import unittest


class SimpleDatasetTest(unittest.TestCase):
    """Tests for SimpleDataset class"""

    def setUp(self):
        num_examples = 10
        self.matrix = numpy.random.random((num_examples, 3)).astype(
            numpy.float32)
        self.labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int16)
        self.indices = {
            'train': [1, 2, 3], 'test': [3, 4, 5], 'val': [0, 9]
        }
        self.dataset = dataset.SimpleDataset()
        self.dataset.create_from_matrixes(self.matrix, indices=self.indices,
                                          labels=self.labels)

    def test_types(self):
        """Test if instances_type and labels_type are correct."""
        self.assertEqual(numpy.float32, self.dataset.instances_type)
        self.assertEqual(numpy.int16, self.dataset.labels_type)

    def test_construct_from_matrixes(self):
        self.assertEqual(self.dataset.indices, self.indices)
        self.assertEqual(len(self.dataset.datasets), len(self.indices))

    def test_unlimited_batches(self):
        """Tests the dataset can produce unlimited batches."""
        for i in range(10):
            batch, labels = self.dataset.next_batch(batch_size=2,
                                                    partition_name='train')
            self.assertEqual(batch.shape[0], 2)
            self.assertEqual(labels.shape[0], 2)
            for row, label in zip(batch, labels):
                instance_index = numpy.where(row == self.matrix)[0]
                self.assertNotEqual(instance_index.shape[0], 0)
                instance_index = instance_index[0]

                self.assertEqual(
                    self.labels[instance_index], label
                )


class SimpleSampledDatasetTest(unittest.TestCase):
    """Test for class SimpleSampledDataset"""
    def setUp(self):
        num_examples = 10
        self.matrix = numpy.random.random((num_examples * 3, 3)).astype(
            numpy.float32)
        self.labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int16)
        # We ensure each label is at least three times
        self.labels = numpy.concatenate([self.labels, self.labels, self.labels])
        self.partition_sizes = {
            'train': 0.5, 'test': 0.25, 'val': 0.1
        }
        self.dataset = dataset.SimpleSampledDataset()
        self.samples = 5
        self.dataset.create_samples(self.matrix, self.labels, self.samples,
                                    self.partition_sizes)

    def test_types(self):
        """Test if instances_type and labels_type are correct."""
        self.dataset.set_current_sample(0)
        self.assertEqual(numpy.float32, self.dataset.instances_type)
        self.assertEqual(numpy.int16, self.dataset.labels_type)

    def test_sample_sizes(self):
        """Test samples have correct proportion"""
        for sample in range(self.samples):
            self.dataset.set_current_sample(sample)
            for partition, proportion in self.partition_sizes.iteritems():
                self.assertEqual(int(proportion * self.matrix.shape[0]),
                                 self.dataset.num_examples(partition))

    def test_unlimited_batches(self):
        """Tests the dataset can produce unlimited batches."""
        self.dataset.set_current_sample(0)
        for i in range(10):
            batch, labels = self.dataset.next_batch(batch_size=2,
                                                    partition_name='train')
            self.assertEqual(batch.shape[0], 2)
            self.assertEqual(labels.shape[0], 2)
            for row, label in zip(batch, labels):
                instance_index = numpy.where(row == self.matrix)[0]
                self.assertNotEqual(instance_index.shape[0], 0)
                instance_index = instance_index[0]

                self.assertEqual(
                    self.labels[instance_index], label
                )


class SequenceDatasetTest(unittest.TestCase):
    """Test for class SimpleSampledDataset"""
    def setUp(self):
        num_examples = 20
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of one element.
        self.matrix = [
            numpy.array([[x, x+1] for x in range(sequence_length)])
            for sequence_length in random.sample(k=num_examples,
                                                 population=range(3, 28))]
        self.matrix = numpy.array(self.matrix)
        self.labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int16)
        # We ensure each label is at least three times
        self.partition_sizes = {
            'train': 0.5, 'test': 0.25, 'val': 0.1
        }
        self.dataset = dataset.SequenceDataset()
        self.dataset.create_samples(self.matrix, self.labels, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)

    def test_types(self):
        """Test if instances_type and labels_type are correct."""
        self.assertIn(self.dataset.instances_type, [numpy.int32, numpy.int64])
        self.assertEqual(numpy.int16, self.dataset.labels_type)

    def test_ordered_instances(self):
        """Test samples have correct proportion"""
        instances = self.dataset._instances
        self.assertTrue(all(len(a) <= len(b) for a, b in
                            zip(instances[:-1], instances[1:])))

    def test_padded_batches(self):
        """Tests the dataset produces batches of padded sentences."""
        for i in range(3):
            batch, labels, lengths = self.dataset.next_batch(
                batch_size=2, partition_name='train')
            batch = batch.astype(self.matrix.dtype)
            self.assertIsInstance(batch, numpy.ndarray)
            self.assertEqual(len(batch.shape), 3)

            self.assertEqual(batch.shape[0], 2)
            self.assertEqual(labels.shape[0], 2)
            self.assertEqual(lengths.shape[0], 2)
            for index, row in enumerate(batch):
                label = labels[index]
                length = lengths[index]
                original_instance = row[:length]

                instance_index = None
                for idx, instance in enumerate(self.matrix):
                    if numpy.array_equal(instance, original_instance):
                        instance_index = idx
                        break
                self.assertIsNotNone(
                    instance_index,
                    msg='Instances {} not present in matrix {}'.format(
                        original_instance, self.matrix))

                self.assertEqual(
                    self.labels[instance_index], label
                )
                if length < batch.shape[1]:
                    self.assertAlmostEqual(0, row[length:].max(), places=3)


class LabeledSequenceDatasetTest(unittest.TestCase):
    """Test for class SimpleSampledDataset"""
    def setUp(self):
        num_examples = 20
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of one element.
        self.matrix = [
            numpy.array([[x, x+1] for x in range(sequence_length)])
            for sequence_length in random.sample(k=num_examples,
                                                 population=range(3, 28))]
        self.matrix = numpy.array(self.matrix)
        self.labels = numpy.array([[element[0] + 4 for element in sequence]
                                   for sequence in self.matrix])
        # We ensure each label is at least three times
        self.partition_sizes = {
            'train': 0.5, 'test': 0.25, 'val': 0.1
        }
        self.dataset = dataset.LabeledSequenceDataset()
        self.dataset.create_samples(self.matrix, self.labels, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)

    def test_padded_batches(self):
        """Tests the dataset produces batches of padded sequences.

        Each padded sequence should be filled with 0s until the nearest
        multiple of max_sequence_length."""
        for i in range(3):
            # We have 20 examples
            batch, labels, lengths = self.dataset.next_batch(
                batch_size=4, partition_name='train', max_sequence_length=3)
            batch = batch.astype(self.matrix.dtype)
            self.assertIsInstance(batch, numpy.ndarray)
            self.assertEqual(len(batch.shape), 3)

            self.assertTrue(batch.shape[1] % 3 == 0)
            self.assertEqual(labels.shape[1], batch.shape[1])

            for index, row in enumerate(batch):
                label = labels[index]
                length = lengths[index]
                original_instance = row[:length]

                instance_index = None
                for idx, instance in enumerate(self.matrix):
                    if numpy.array_equal(instance, original_instance):
                        instance_index = idx
                        break
                self.assertIsNotNone(
                    instance_index,
                    msg='Instances {} not present in matrix {}'.format(
                        original_instance, self.matrix))

                self.assertTrue(numpy.array_equal(
                    self.labels[instance_index], label[:length])
                )
                if length < batch.shape[1]:
                    self.assertAlmostEqual(0, row[length:].max(), places=3)


class TestUnlabeledSequenceDataset(unittest.TestCase):

    def _get_one_hot_encoding(self, x):
        result = numpy.zeros(self.feature_size)
        result[x] = 1
        return result

    def _get_random_sequence(self):
        return numpy.array([
            self._get_one_hot_encoding(x % self.feature_size)
            for x in range(random.randint(3, self.max_num_steps))])

    def setUp(self):
        num_examples = 50
        self.feature_size = 5
        self.max_num_steps = 20
        self.matrix = [self._get_random_sequence() for _ in range(num_examples)]
        self.matrix = numpy.array(self.matrix)

        self.partition_sizes = {
            'train': 0.65, 'test': 0.25, 'validation': 0.1
        }
        self.dataset = dataset.UnlabeledSequenceDataset()
        self.dataset.create_samples(self.matrix, None, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)

    def test_get_labels(self):
        labels = self.dataset._get_labels()
        self.assertEqual(labels.shape[0],
                         self.dataset._instances.shape[0])
        for sequence, sequence_labels in zip(self.dataset._instances, labels):
            # Same number of labels and elements in the sequence
            self.assertEqual(sequence.shape[0], sequence_labels.shape[0])
            self.assertEqual(1, sequence_labels.ndim)
            for index in range(sequence.shape[0] - 1):
                # The label must be equal to the index in the next one hot
                # encoding.
                self.assertEqual(numpy.argmax(sequence[index + 1]),
                                 sequence_labels[index])
            # Check last label
            self.assertEqual(sequence_labels[-1], self.dataset.EOS_symbol)


if __name__ == '__main__':
    unittest.main()
