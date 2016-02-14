#!/usr/bin/python3

from __future__ import print_function

from main import run
from layers import *

network = compose(
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
	xaffine(128, 512),
	bnorm(512, 0.1),
	relu(),
	maxout(512, 512, 4),
	xaffine(512, 10),
	relu(),
	softmax()
	)

run(network, "mark8")
