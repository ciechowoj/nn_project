#!/usr/bin/python3

from __future__ import print_function

from main import run
from layers import *

network = compose(
    conv2D(3, 128, 5), 
    relu(), 
    max_pool_2d(2),
    conv2D(128, 128, 5),
    bnorm2D(128, 0.1),
    relu(), 
    max_pool_2d(2),
    flatten(),
    xaffine(3200, 625),
    bnorm(625, 0.1),
    relu(),
    xaffine(625, 625),
    bnorm(625, 0.1),
    relu(),
    xaffine(625, 10),
    relu(),
    softmax()
    )

run(network, "mark3c")
