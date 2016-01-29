from __future__ import print_function

import theano
import numpy
import random
import theano.tensor as tensor
from theano.ifelse import ifelse

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
    
    def fprop(X, test):
        return theano.tensor.nnet.conv2d(X, weights) + biases.dimshuffle('x', 0, 'x', 'x')
    
    fprop.params = [weights, biases]
    
    return fprop

def relu(name = fresh_name("?")):
    def fprop(X, test):
        return theano.tensor.maximum(0.0, X)
    
    return fprop
    
def max_pool_2d(kernel_size):
    def fprop(X, test):
        kernel_shape = (kernel_size, kernel_size)
        return theano.tensor.signal.downsample.max_pool_2d(X, kernel_shape, ignore_border = True)

    return fprop

def flatten(name = None):
    def fprop(X, test):
        return X.flatten(2)

    return fprop

def dropout(prob = 0.5, name = None):
    rng = theano.tensor.shared_randomstreams.RandomStreams(random.randint(1, 748978023))
    rprop = 1 - prob
    
    def fprop(X, test):
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
    
    def fprop(X, test):
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
    
    def fprop(X, test):
        return theano.tensor.max(theano.tensor.dot(X, weights) + biases.dimshuffle('x', 0, 1), axis = 2)
    
    fprop.params = [weights, biases]
    
    return fprop
    
def bnorm(num_inputs, alpha, epsilon = 1e-4, name = None):
    name = name if name else fresh_name("?")

    def make_vparam(pname, c):
        shape = (num_inputs,)
        xname = "{}.{}".format(name, pname)
        param = theano.shared(numpy.zeros(shape, dtype='float32'), name = xname)
        param.tag.initializer = Constant(c)
        return param

    gammas = make_vparam("gammas", 1)
    betas = make_vparam("betas", 0)
    means = make_vparam("mean_avgs", 0)
    inv_stds = make_vparam("inv_std_avgs", 0)

    def lerp(x, y, a):
        return (1 - a) * x + a * y

    def ds(v):
        return v.dimshuffle('x', 0)

    def fprop(X, test):
        btest = tensor.lt(0, test)

        X_means = X.mean(0)
        X_inv_stds = tensor.inv(tensor.sqrt(X.var(0)) + epsilon)

        means_clone = theano.clone(means, share_inputs = False)
        inv_stds_clone = theano.clone(inv_stds, share_inputs = False)

        means_clone.default_update = ifelse(btest, means, lerp(means, X_means, alpha))
        inv_stds_clone.default_update = ifelse(btest, inv_stds, lerp(inv_stds, X_inv_stds, alpha))
    
        X_means += 0 * means_clone
        X_inv_stds += 0 * inv_stds_clone

        X_means = ifelse(btest, means, X_means)
        X_inv_stds = ifelse(btest, inv_stds, X_inv_stds)

        return (X - ds(X_means)) * ds(X_inv_stds) * ds(gammas) + ds(betas)

    fprop.params = [gammas, betas]
    fprop.variables = [means, inv_stds]

    return fprop

def bnorm2D(num_input_filters, alpha, epsilon = 1e-4, name = None):
    name = name if name else fresh_name("?")

    def make_vparam(pname, c):
        shape = (num_input_filters,)
        xname = "{}.{}".format(name, pname)
        param = theano.shared(numpy.zeros(shape, dtype='float32'), name = xname)
        param.tag.initializer = Constant(c)
        return param

    gammas = make_vparam("gammas", 1)
    means = make_vparam("mean_avgs", 0)
    inv_stds = make_vparam("inv_std_avgs", 0)

    def lerp(x, y, a):
        return (1 - a) * x + a * y

    def ds(v):
        return v.dimshuffle('x', 0, 'x', 'x')

    def fprop(X, test):
        btest = tensor.lt(0, test)

        X_means = X.mean([0, 2, 3])
        X_inv_stds = tensor.inv(tensor.sqrt(X.var([0, 2, 3])) + epsilon)

        means_clone = theano.clone(means, share_inputs = False)
        inv_stds_clone = theano.clone(inv_stds, share_inputs = False)

        means_clone.default_update = ifelse(btest, means, lerp(means, X_means, alpha))
        inv_stds_clone.default_update = ifelse(btest, inv_stds, lerp(inv_stds, X_inv_stds, alpha))
    
        X_means += 0 * means_clone
        X_inv_stds += 0 * inv_stds_clone

        X_means = ifelse(btest, means, X_means)
        X_inv_stds = ifelse(btest, inv_stds, X_inv_stds)

        return (X - ds(X_means)) * ds(X_inv_stds) * ds(gammas)

    fprop.params = [gammas]
    fprop.variables = [means, inv_stds]

    return fprop

def softmax():
    def fprop(X, test):
        return theano.tensor.nnet.softmax(X)

    return fprop
    
def compose(*args):
    def fprop(X, test):
        for arg in args:
            X = arg(X, test)
        return X
    
    params = []
    variables = []

    for arg in args:
        try:
            params += arg.params
        except AttributeError:
            pass

        try:
            variables += arg.variables
        except AttributeError:
            pass
    
    if params != []:
        fprop.params = params

    if variables != []:
        fprop.variables = variables
    
    return fprop
