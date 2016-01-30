#!/usr/bin/python3

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

from prepare_cifar10 import *

cifar = prepare_cifar10()

cifar_train = cifar.train
cifar_train_stream = cifar.train_stream
											   
cifar_validation = cifar.validation
cifar_validation_stream = cifar.validation_stream

cifar_test = cifar.test
cifar_test_stream = cifar.test_stream

nn = compose(
	conv2D(3, 128, 3),		# 31 x 31 
	relu(), 
	conv2D(128, 128, 3),	# 30 x 30
	bnorm2D(128, 0.1),		
	relu(), 
	max_pool_2d(2),			# 15 x 15
	conv2D(128, 128, 3),	# 14 x 14
	bnorm2D(128, 0.1),		
	relu(), 
	max_pool_2d(2),
	conv2D(128, 128, 5),
	flatten(),
	xaffine(512, 512),
	bnorm(512, 0.1),
	relu(),
	xaffine(512, 512),
	bnorm(512, 0.1),
	relu(),
	xaffine(512, 10),
	relu(),
	softmax()
	)

print("Compiling...", end = " ")
sys.stdout.flush()

network = compile(nn)

print("DONE")
sys.stdout.flush()

train(network, cifar_train_stream, cifar_validation_stream, 4e-3, 0.7)

print("Test error rate is %f%%" %(compute_error_rate(cifar_test_stream, network.predict) * 100.0,))
