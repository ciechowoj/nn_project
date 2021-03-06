#!/usr/bin/python3

from __future__ import print_function

from main import run
from layers import *

network = compose(
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
	xaffine(512, 512),
	bnorm(512, 0.1),
	relu(),
	xaffine(512, 512),
	bnorm(512, 0.1),
	relu(),
	maxout(512, 64, 8),
	xaffine(64, 10),
	relu(),
	softmax()
	)

run(network, "mark2b")
