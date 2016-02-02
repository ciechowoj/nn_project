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
	max_pool_2d(2),			# 7 x 7
	conv2D(128, 128, 5),    # 5 x 5
	flatten(),
	xaffine(512, 512),
	bnorm(512, 0.1),
	relu(),
	maxout(512, 512, 4),
	xaffine(512, 10),
	relu(),
	softmax()
	)

print("Compiling network...", end = " ")
sys.stdout.flush()
network = compile(nn)
print("DONE")
sys.stdout.flush()

print("Loading parameters...", end = " ")
sys.stdout.flush()
network.load("mark8.fn")
print("DONE")
sys.stdout.flush()

print("Computing test error rate...", end = " ")
sys.stdout.flush()
error_rate = compute_error_rate(
	cifar_test_stream, 
	network.predict,
	"\rComputing test error rate... {}%")
print("\rComputing test error rate... DONE")
sys.stdout.flush()

print("Test error rate is {:.2f}%.".format(error_rate * 100.0,))
