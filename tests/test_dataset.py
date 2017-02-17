import dataset
import numpy
import unittest


class SimpleDatasetTest(unittest.TestCase):
    """Tests for SimpleDataset class"""

    def setUp(self):
        num_examples = 10
        self.matrix = numpy.random.random((num_examples, 3))
        self.labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int16)
        self.indices = {
            'train': [1, 2, 3], 'test': [3, 4, 5], 'val': [0, 9]
        }
        self.dataset = dataset.SimpleDataset()
        self.dataset.create_from_matrixes(self.matrix, indices=self.indices,
                                          labels=self.labels)

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
        self.matrix = numpy.random.random((num_examples * 3, 3))
        self.labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int16)
        # We ensure each label is at least three times
        self.labels = numpy.concatenate([self.labels, self.labels, self.labels])
        self.partition_sizes = {
            'train': 0.5, 'test': 0.25, 'val': 0.1
        }
        self.dataset = dataset.SimpleSampledDataset()

    def test_sample_sizes(self):
        """Test samples have correct proportion"""
        samples = 5
        self.dataset.create_samples(self.matrix, self.labels, samples,
                                    self.partition_sizes)
        for sample in range(samples):
            self.dataset.set_current_sample(sample)
            for partition, proportion in self.partition_sizes.iteritems():
                self.assertEqual(int(proportion * self.matrix.shape[0]),
                                 self.dataset.num_examples(partition))

    def test_unlimited_batches(self):
        """Tests the dataset can produce unlimited batches."""
        self.dataset.create_samples(self.matrix, self.labels, 5,
                                    self.partition_sizes)
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


if __name__ == '__main__':
    unittest.main()