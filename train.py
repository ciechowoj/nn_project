
from __future__ import print_function

import theano
import theano.tensor.signal.downsample
from utils import *
import time
import pickle
import sys

from layers import *
from network import *
from train import *

def train(network, test_stream, validation_stream, learn_rate0, momentum, path):
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

	best_valid_error_rate = numpy.inf
	best_params = network.snapshot()
	best_params_epoch = 0

	# training loop
	try:
		start = time.time()
		
		while not quit and epoch < number_of_epochs: #This loop goes over epochs
			epoch += 1
			#First train on all data from this batch

			epoch_start_batch = batch

			for X_batch, Y_batch in test_stream.get_epoch_iterator(): 
				batch += 1

				K = 100000
				learn_rate = learn_rate0 * K / numpy.maximum(K, batch)
				
				L, err_rate, nll, wdec = network.step(X_batch, Y_batch, learn_rate, momentum)

				train_loss.append((batch, L))
				train_erros.append((batch, err_rate))
				train_nll.append((batch, nll))

			# After an epoch compute validation error
			val_error_rate = compute_error_rate(validation_stream, network.predict)
			if val_error_rate < best_valid_error_rate:
				number_of_epochs = numpy.maximum(number_of_epochs, int(epoch * patience_expansion + 1))
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
		network.dump("{}_{}".format(path, time.strftime("%d_%H_%M_%S")), records)
		raise

	print("Setting network parameters from after epoch %d" %(best_params_epoch))
	network.load(best_params)
	network.dump("{}_{}".format(path, time.strftime("%d_%H_%M_%S")), records)
