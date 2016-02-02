from __future__ import print_function

import theano
import theano.tensor.signal.downsample
from utils import *
import time
import pickle
import json
import sys

from layers import *
from network import *
from train import *

class Network:
	def snapshot(self):
		params = [p.get_value(borrow = False).tolist() for p in self.params]
		velocities = [v.get_value(borrow = False).tolist() for v in self.velocities]
		variables = [v.get_value(borrow = False).tolist() for v in self.variables]

		result = {
			"params" : params,
			"velocities" : velocities,
			"variables" : variables
		}

		return result

	def _load(self, source):
			try:
				with open(source, 'rb') as file:
					snapshot, records = pickle.load(file)

					snapshot = {
						"params" : snapshot[0],
						"velocities" : snapshot[1],
						"variables" : snapshot[2]
					}

					self.load(snapshot, False)
					return records
			except KeyError:
				pass
			except pickle.UnpicklingError:
				pass

			with open("{}.network".format(path), 'r') as file:
				data = json.load(file)
				self.load(data, False)

			with open("{}.records".format(path), 'r') as file:
				data = json.load(file)
				return data

	def load(self, source, file = True):
		if file and isinstance(source, str):
			return self._load(source)
		else:
			for p, s in zip(self.params, source["params"]):
				p.set_value(s, borrow = False)

			for v, s in zip(self.velocities, source["velocities"]):
				v.set_value(s, borrow = False)

			for v, s in zip(self.variables, source["variables"]):
				v.set_value(s, borrow = False)

	def dump(self, path, records = None):
		with open("{}.network".format(path), 'w+') as file:
			json.dump(self.snapshot(), file, sort_keys=True, indent=4, separators=(',', ': '))

		with open("{}.records".format(path), 'w+') as file:
			json.dump(records, file, sort_keys=True, indent=4, separators=(',', ': '))			

def compile(template):
	X = theano.tensor.tensor4('X', dtype = 'float32')
	Y = theano.tensor.matrix('Y', dtype = 'uint8')
	
	model_parameters = template.params

	test = theano.tensor.scalar('test', dtype = 'float32')

	log_probs = template(X, test)

	predictions = theano.tensor.argmax(log_probs, axis = 1)

	error_rate = theano.tensor.neq(predictions,Y.ravel()).mean()
	nll = - theano.tensor.log(log_probs[theano.tensor.arange(Y.shape[0]), Y.ravel()]).mean()

	weight_decay = 0.0
	for p in model_parameters:
		if p.name.endswith('weights'):
			weight_decay = weight_decay + 1e-3 * (p ** 2).sum()

	cost = nll + weight_decay
	
	learn_rate = theano.tensor.scalar('learn_rate', dtype = 'float32')
	momentum = theano.tensor.scalar('momentum', dtype = 'float32')

	# Theano will compute the gradients for us
	gradients = theano.grad(cost, model_parameters)

	#initialize storage for momentum
	velocities = [theano.shared(numpy.zeros_like(p.get_value()), name='V_%s' %(p.name, )) for p in model_parameters]
	
	updates = []

	for p, g, v in zip(model_parameters, gradients, velocities):
		v_new = momentum * v - learn_rate * g
		p_new = p + v_new
		updates += [(v, v_new), (p, p_new)]
	
	def init_parameters():
		rng = numpy.random.RandomState(1234)
		
		for p in model_parameters:
			p.set_value(p.tag.initializer.generate(rng, p.get_value().shape))
			
		for v in velocities:
			v.set_value(numpy.zeros_like(v.get_value()))
			
	step = theano.function(
		[X, Y, learn_rate, momentum, test],
		[cost, error_rate, nll, weight_decay],
		updates = updates, 
		allow_input_downcast = True,
		on_unused_input = 'warn')
			
	predict = theano.function([X, test], predictions)
	
	init_parameters()
	
	def step_ex(X, Y, learn_rate, momentum):
		return step(X, Y, learn_rate, momentum, 0)

	def predict_ex(X):
		return predict(X, 1)

	network = Network()
	network.step = step_ex
	network.predict = predict_ex
	network.params = model_parameters
	network.velocities = velocities
	network.variables = template.variables

	return network
