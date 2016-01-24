from __future__ import print_function

from fuel.datasets.cifar10 import CIFAR10
from fuel.transformers import ScaleAndShift, Cast, Flatten, Mapping
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
import numpy as np
import numpy
import types


def cifar10_mean():
	train = CIFAR10(("train",), subset=slice(None, 40000))
	train_stream = DataStream.default_stream(train, iteration_scheme = SequentialScheme(train.num_examples, 100))

	X = numpy.array([numpy.mean(X, 0) for X, _ in train_stream.get_epoch_iterator()])
	X = numpy.mean(X, 0)

	return X

def prepare_cifar10():
	class Dataset:
		pass

	result = Dataset()

	CIFAR10.default_transformers = (
		(ScaleAndShift, [2.0 / 255.0, -1], {'which_sources': 'features'}),
		(Cast, [np.float32], {'which_sources': 'features'}))

	mean = cifar10_mean()

	def patch_get_epoch_iterator(stream):
		def get_epoch_iterator(self):
			for X, Y in self._get_epoch_iterator():
				X -= mean[numpy.newaxis,:,:,:]
				yield X, Y

		stream._get_epoch_iterator = stream.get_epoch_iterator
		stream.get_epoch_iterator = types.MethodType(get_epoch_iterator, stream)

	result.train = train = CIFAR10(("train",), subset = slice(None, 40000))
	result.train_stream = DataStream.default_stream(
		result.train,
		iteration_scheme = ShuffledScheme(result.train.num_examples, 25))

	patch_get_epoch_iterator(result.train_stream)

	result.validation = CIFAR10(("train",), subset=slice(40000, None))
	result.validation_stream = DataStream.default_stream(
		result.validation, 
		iteration_scheme = SequentialScheme(result.validation.num_examples, 100))

	patch_get_epoch_iterator(result.validation_stream)

	result.test = CIFAR10(("test",))
	result.test_stream = DataStream.default_stream(
		result.test, 
		iteration_scheme = SequentialScheme(result.test.num_examples, 100))

	patch_get_epoch_iterator(result.test_stream)

	return result