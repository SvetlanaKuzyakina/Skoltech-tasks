import numpy as np
from numba import njit, prange

@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64,int32[:],int32,int32)")
def generate_kernels(input_length, num_kernels, candidate_lengths, bias_type, weight_type): 
    # time series length, number of kernels, kernel dimensions, bias type (N(0,1) or U[-1,1]),
    # weight type (N(0,1) or Binary([-1,1]) or Trinary([-1,0,1]))

    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    if weight_type == 2: # binary weights
        weight_values = np.array((-1, 1), dtype = np.float64)
    elif weight_type == 3: # trinary weights
        weight_values = np.array((-1, 0, 1), dtype = np.float64)
    
    a1 = 0

    for i in range(num_kernels):
        _length = lengths[i]
        
        if weight_type == 2 or weight_type == 3:
            _weights = np.random.choice(weight_values, _length)
        else:    
            _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        if bias_type == 2: 
            biases[i] = np.random.normal(0, 1)
        else:
            biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    _ppv = 0
    _max = np.NINF
    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias
        index = i

        for j in range(length):
            if index > -1 and index < input_length:
                _sum = _sum + weights[j] * X[index]
            index = index + dilation

        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])),int32)", parallel = True, fastmath = True)
def apply_kernels(X, kernels, only_ppv): # only_ppv: 0 - extract ppv and max from each kernel, 1 - extract only ppv
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    
    if only_ppv == 1: # extract only ppv for each kernel
        _X = np.zeros((num_examples, num_kernels), dtype = np.float64) 

        for i in prange(num_examples):
            a1 = 0 # for weights
            a2 = 0 # for features
            
            for j in range(num_kernels):
                b1 = a1 + lengths[j]
                b2 = a2 + 1 # one feature - ppv

                _X[i, a2:b2] = \
                apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j]) [0]

                a1 = b1
                a2 = b2

        return _X
        
    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):
        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2 # two features - ppv and max

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X


