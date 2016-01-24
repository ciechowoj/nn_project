from __future__ import print_function

import numpy

def fresh_name(name):
    D = {}
    try:
        D = fresh_name.D
    except AttributeError:
        fresh_name.D = {}
        D = fresh_name.D
        
    if name in D:
        D[name] += 1
    else:
        D[name] = 1
        
    return name + str(D[name])

def print_hline():
        cell_width = 20
        print("|{}|{}|{}|{}|{}|".format(
            "-" * (cell_width + 2), 
            "-" * (cell_width - 1), 
            "-" * (cell_width - 1), 
            "-" * cell_width, 
            "-" * cell_width))

def print_header():
    print("|{:^22}|{:^19}|{:^19}|{:^20}|{:^20}|".format(
        "epoch//batch",
        "validation error",
        "avg. train error",
        "avg. train nll",
        "avg. train loss"))

def print_record(record):
    print("|{:^21} |{:>18.2f} |{:>18.2f} |{:>19g} |{:>19g} |".format(*record))
    print_hline()
    sys.stdout.flush()

def compute_error_rate(stream, predict):
    errs = 0.0
    num_samples = 0.0
    for X, Y in stream.get_epoch_iterator():
        errs += (predict(X) != Y.ravel()).sum()
        num_samples += Y.shape[0]
    return errs / num_samples

#
# These are taken from https://github.com/mila-udem/blocks
# 

class Constant():
    """Initialize parameters to a constant.
    The constant may be a scalar or a :class:`~numpy.ndarray` of any shape
    that is broadcastable with the requested parameter arrays.
    Parameters
    ----------
    constant : :class:`~numpy.ndarray`
        The initialization value to use. Must be a scalar or an ndarray (or
        compatible object, such as a nested list) that has a shape that is
        broadcastable with any shape requested by `initialize`.
    """
    def __init__(self, constant):
        self._constant = numpy.asarray(constant)

    def generate(self, rng, shape):
        dest = numpy.empty(shape, dtype = numpy.float32)
        dest[...] = self._constant
        return dest


class IsotropicGaussian():
    """Initialize parameters from an isotropic Gaussian distribution.
    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0
    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!
    """
    def __init__(self, std=1, mean=0):
        self._mean = mean
        self._std = std

    def generate(self, rng, shape):
        m = rng.normal(self._mean, self._std, size=shape)
        return m.astype(numpy.float32)


class Uniform():
    """Initialize parameters from a uniform distribution.
    Parameters
    ----------
    mean : float, optional
        The mean of the uniform distribution (i.e. the center of mass for
        the density function); Defaults to 0.
    width : float, optional
        One way of specifying the range of the uniform distribution. The
        support will be [mean - width/2, mean + width/2]. **Exactly one**
        of `width` or `std` must be specified.
    std : float, optional
        An alternative method of specifying the range of the uniform
        distribution. Chooses the width of the uniform such that random
        variates will have a desired standard deviation. **Exactly one** of
        `width` or `std` must be specified.
    """
    def __init__(self, mean=0., width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, "
                             "but not both")
        if std is not None:
            # Variance of a uniform is 1/12 * width^2
            self._width = numpy.sqrt(12) * std
        else:
            self._width = width
        self._mean = mean

    def generate(self, rng, shape):
        w = self._width / 2
        m = rng.uniform(self._mean - w, self._mean + w, size=shape)
        return m.astype(numpy.float32)

