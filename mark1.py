#!/usr/bin/python3

from __future__ import print_function

from main import run
from layers import *

network = compose(
    conv2D(3, 64, 5), 
    relu(), 
    max_pool_2d(2),
    conv2D(64, 128, 5), 
    relu(), 
    max_pool_2d(2),
    flatten(),
    xaffine(3200, 625),
    relu(),
    xaffine(625, 10),
    relu(),
    softmax()
    )

run(network, "mark1")
