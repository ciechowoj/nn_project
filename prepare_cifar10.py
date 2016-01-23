from fuel.datasets.cifar10 import CIFAR10
from fuel.transformers import ScaleAndShift, Cast, Flatten, Mapping
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
import numpy as np

def prepare_cifar10():
	class Dataset:
		pass

	result = Dataset()

	CIFAR10.default_transformers = (
    	(ScaleAndShift, [2.0 / 255.0, -1], {'which_sources': 'features'}),
    	(Cast, [np.float32], {'which_sources': 'features'}))

	result.train = train = CIFAR10(("train",), subset=slice(None,40000))
	result.train_stream = DataStream.default_stream(
	    result.train,
	    iteration_scheme = ShuffledScheme(result.train.num_examples, 100))
	                                               
	result.validation = CIFAR10(("train",), subset=slice(40000, None))
	result.validation_stream = DataStream.default_stream(
	    result.validation, 
	    iteration_scheme = SequentialScheme(result.validation.num_examples, 100))

	result.test = CIFAR10(("test",))
	result.test_stream = DataStream.default_stream(
	    result.test, 
	    iteration_scheme = SequentialScheme(result.test.num_examples, 100))

	return result
