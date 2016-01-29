#!/usr/bin/python3

from __future__ import print_function

import theano
import theano.tensor.signal.downsample
from utils import *
import time
import pickle
import sys

from layers import *

from prepare_cifar10 import *

cifar = prepare_cifar10()

cifar_train = cifar.train
cifar_train_stream = cifar.train_stream
											   
cifar_validation = cifar.validation
cifar_validation_stream = cifar.validation_stream

cifar_test = cifar.test
cifar_test_stream = cifar.test_stream

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
	velocities = [theano.shared(np.zeros_like(p.get_value()), name='V_%s' %(p.name, )) for p in model_parameters]
	
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
			v.set_value(np.zeros_like(v.get_value()))
			
	step = theano.function(
		[X, Y, learn_rate, momentum, test],
		[cost, error_rate, nll, weight_decay],
		updates = updates, 
		allow_input_downcast = True,
		on_unused_input = 'warn')
			
	predict = theano.function([X], predictions)
	
	init_parameters()
	
	def step_ex(X, Y, learn_rate, momentum):
		return step(X, Y, learn_rate, momentum, 0)

	def predict_ex(X):
		return predict(X, 1)

	class Network:
		def snapshot(self):
			return ([p.get_value(borrow = False) for p in self.params], [v.get_value(borrow = False) for v in self.velocities])
	
		def load(self, source, file = True):
			if file and isinstance(source, str):
				with open(source, 'rb') as file:
					snapshot, records = pickle.load(file)
					self.load(snapshot, False)
					return records
			else:
				for p, s in zip(self.params, source[0]):
					p.set_value(s, borrow = False)

				for v, s in zip(self.velocities, source[1]):
					v.set_value(s, borrow = False)
	
		def dump(self, path, records = None):
			with open(path, 'wb+') as file:
				pickle.dump((self.snapshot(), records), file)

	network = Network()
	network.step = step_ex
	network.predict = predict_ex
	network.params = model_parameters
	network.velocities = velocities

	return network

def train(network, learn_rate0, momentum):
	batch = 0
	epoch = 0
	epoch_offset = 0

	train_erros = []
	train_loss = []
	train_nll = []
	validation_errors = []

	number_of_epochs = 3
	patience_expansion = 2

	print_hline()
	print_header()
	print_hline()

	quit = False

	records = []
	if len(sys.argv) == 2:
		records = network.load(sys.argv[1])

	for record in records:
		print_record(record)

	try:
		epoch, number_of_epochs, batch = [int(s.strip()) for s in records[-1][0].split('/')]
		epoch_offset = batch
	except:
		epoch, number_of_epochs, batch = (0, 3, 0)

	best_valid_error_rate = np.inf
	best_params = network.snapshot()
	best_params_epoch = 0

	# training loop
	try:
		start = time.time()
		
		while not quit and epoch < number_of_epochs: #This loop goes over epochs
			epoch += 1
			#First train on all data from this batch

			epoch_start_batch = batch

			for X_batch, Y_batch in cifar_train_stream.get_epoch_iterator(): 
				batch += 1

				# learn_rate = learn_rate0 * (1 - np.tanh(batch * 0.549306 / 2000))
				
				K = 100000
				learn_rate = learn_rate0 * K / np.maximum(K, batch)
				
				L, err_rate, nll, wdec = network.step(X_batch, Y_batch, learn_rate, momentum)

				train_loss.append((batch, L))
				train_erros.append((batch, err_rate))
				train_nll.append((batch, nll))

			# After an epoch compute validation error
			val_error_rate = compute_error_rate(cifar_validation_stream, network.predict)
			if val_error_rate < best_valid_error_rate:
				number_of_epochs = np.maximum(number_of_epochs, int(epoch * patience_expansion + 1))
				best_valid_error_rate = val_error_rate
				best_params = network.snapshot()
				best_params_epoch = epoch
			validation_errors.append((batch, val_error_rate))

			record = (
				"{} / {} / {}".format(epoch, number_of_epochs, batch), 
				val_error_rate * 100, 
				numpy.mean(numpy.asarray(train_erros)[epoch_start_batch - epoch_offset:, 1]) * 100, 
				numpy.mean(numpy.asarray(train_nll)[epoch_start_batch - epoch_offset:, 1]),
				numpy.mean(numpy.asarray(train_loss)[epoch_start_batch - epoch_offset:, 1]))

			records.append(record)
			print_record(record)
			

	except KeyboardInterrupt:
		pass
	except:
		network.dump("{}.nn".format(time.strftime("%d_%b_%Y_%H_%M_%S")), records)
		raise

	print("Setting network parameters from after epoch %d" %(best_params_epoch))
	network.load(best_params)
	network.dump("{}.nn".format(time.strftime("%d_%b_%Y_%H_%M_%S")), records)

	#subplot(2,1,1)
	#train_nll_a = np.array(train_nll)
	#semilogy(train_nll_a[:,0], train_nll_a[:,1], label='batch train nll')
	#legend()

	#subplot(2,1,2)
	#train_erros_a = np.array(train_erros)
	#plot(train_erros_a[:,0], train_erros_a[:,1], label='batch train error rate')
	#validation_errors_a = np.array(validation_errors)
	#plot(validation_errors_a[:,0], validation_errors_a[:,1], label='validation error rate', color='r')
	#ylim(0,0.2)
	#legend()

nn = compose(
	conv2D(3, 128, 3), 
	relu(), 
	max_pool_2d(2),
	conv2D(128, 128, 3), 
	relu(), 
	max_pool_2d(2),
	conv2D(128, 128, 3), 
	relu(), 
	max_pool_2d(2),
	flatten(),
	xaffine(512, 625),
	relu(),
	xaffine(625, 10),
	relu(),
	softmax()
	)

print("Compiling...", end = " ")
sys.stdout.flush()

network = compile(nn)

print("DONE")
sys.stdout.flush()

train(network, 4e-3, 0.7)

print("Test error rate is %f%%" %(compute_error_rate(cifar_test_stream, network.predict) * 100.0,))
