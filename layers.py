from __future__ import print_function

import theano
import numpy
import random

from utils import *

def conv2D(num_in_filters, num_out_filters, kernel_size, name = None):
    name = name if name else fresh_name("?")
        
    weights_shape = (num_out_filters, num_in_filters, kernel_size, kernel_size)
    weights_name = "{}.weights".format(name)
    weights = theano.shared(numpy.zeros(weights_shape, dtype = 'float32'), name = weights_name)
    weights.tag.initializer = IsotropicGaussian(0.05)
    
    biases_shape = (num_out_filters,)
    biases_name = "{}.biases".format(name)
    biases = theano.shared(numpy.zeros(biases_shape, dtype='float32'), biases_name)
    biases.tag.initializer = Constant(0.0)
    
    def fprop(X):
        return theano.tensor.nnet.conv2d(X, weights) + biases.dimshuffle('x', 0, 'x', 'x')
    
    fprop.params = [weights, biases]
    
    return fprop

def relu(name = fresh_name("?")):
    def fprop(X):
        return theano.tensor.maximum(0.0, X)
    
    return fprop
    
def max_pool_2d(kernel_size):
    def fprop(X):
        kernel_shape = (kernel_size, kernel_size)
        return theano.tensor.signal.downsample.max_pool_2d(X, kernel_shape, ignore_border = True)

    return fprop

def flatten(name = None):
    def fprop(X):
        return X.flatten(2)

    return fprop

def dropout(prob = 0.5, name = None):
    rng = theano.tensor.shared_randomstreams.RandomStreams(random.randint(1, 748978023))
    rprop = 1 - prob
    
    def fprop(X):
        return X * rng.binomial(X.shape, p = rprop, dtype = "float32")
    
    return fprop
    
def xaffine(num_inputs, num_outputs, name = None):    
    name = name if name else fresh_name("?")
    
    weights_shape = (num_inputs, num_outputs)
    weights_name = "{}.weights".format(name)
    weights = theano.shared(numpy.zeros(weights_shape, dtype='float32'), name = weights_name)
    weights.tag.initializer = IsotropicGaussian(0.05)
    
    biases_shape = (num_outputs, )
    biases_name = "{}.biases".format(name)
    biases = theano.shared(numpy.zeros(biases_shape, dtype='float32'), name = biases_name)
    biases.tag.initializer = Constant(0.0)
    
    def fprop(X):
        return theano.tensor.dot(X, weights) + biases.dimshuffle('x', 0)

    fprop.params = [weights, biases]
    
    return fprop
    
def maxout(num_inputs, num_outputs, degree, name = None):
    name = name if name else fresh_name("?")
    
    weights_shape = (num_outputs, num_inputs, degree)
    weights_name = "{}.weights".format(name)
    weights = theano.shared(numpy.zeros(weights_shape, dtype='float32'), name = weights_name)
    weights.tag.initializer = IsotropicGaussian(0.05)
    
    biases_shape = (num_outputs, degree)
    biases_name = "{}.biases".format(name)
    biases = theano.shared(numpy.zeros(biases_shape, dtype='float32'), name = biases_name)
    biases.tag.initializer = Constant(0.0)
    
    def fprop(X):
        return theano.tensor.max(theano.tensor.dot(X, weights) + biases.dimshuffle('x', 0, 1), axis = 2)
    
    fprop.params = [weights, biases]
    
    return fprop
    
def softmax():
    def fprop(X):
        return theano.tensor.nnet.softmax(X)

    return fprop
    
def compose(*args):
    def fprop(X):
        for arg in args:
            X = arg(X)
        return X
    
    params = []

    for arg in args:
        try:
            params += arg.params
        except AttributeError:
            pass
    
    if params != []:
        fprop.params = params
    
    return fprop
