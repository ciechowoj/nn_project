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
	relu(),
	xaffine(512, 512),
	relu(),
	xaffine(512, 10),
	relu(),
	softmax()
	)

run(network, "mark2a")
