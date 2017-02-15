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
        self.dat = dataset.SimpleDataset()
        self.dat.create_from_matrixes(self.matrix, indices=self.indices,
                                      labels=self.labels)

    def test_construct_from_matrixes(self):
        self.assertEqual(self.dat.indices, self.indices)
        self.assertEqual(len(self.dat.datasets), len(self.indices))

    def test_unlimited_batches(self):
        for i in range(10):
            batch, labels = self.dat.next_batch(batch_size=2,
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


# class SimpleSampledDatasetTest(unittest.TestCase):



if __name__ == '__main__':
    unittest.main()