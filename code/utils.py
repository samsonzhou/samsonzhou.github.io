import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib

def hash_func(seed, x, k, p=2**31 - 1):
    '''
     Description:
        Produces hash of input params
    Parameters:
        seed: seed for random generation control
        x: input parameter
        k: size of output space
        p: constant used in hash
    Outputs:
        hash
    '''
    rng = np.random.default_rng(seed)
    coeff = [rng.integers(0, p) for _ in range(4)]
    x = int(x)
    poly = 0
    for i in range(4):
        term = (coeff[i]*pow(x, i, p)) % p
        poly = (poly + term) % p
    return poly % k

def seeded_input_random(seed, input_val):
    h = hashlib.sha256(f"{seed}:{input_val}".encode()).digest()
    return random.Random(int.from_bytes(h, 'big')).random()

def generate_stream_uniform(n, m, seed=42):
    '''
        Description:
            generates a random stream and its frequency in a uniform manner
        Inputs:
            n: size of frequency vector
            m: size of stream
            seed: random seed|
        Returns:
            stream: stream vector of random value
            frequency: frequency vector corresponding to stream vector
    '''

    ## generate a random stream vector
    np.random.seed(seed)
    stream = np.random.randint(1, n+1, size=m)
    ## create the corresponding frequency vector
    frequency = np.zeros(n)
    for s_i in stream:
        frequency[s_i-1] = frequency[s_i-1] + 1
    return stream, frequency

def generate_stream_skewed(n, m, q, k, seed=42):
    '''
        Description:
            generates a random stream and its frequency in a skewed manner
        Inputs:
            n: size of frequency vector
            m: size of stream
            q: desired number of heavy hitters
            k: probability amplifier
            seed: random seed
        Returns:
            stream: stream vector of skewed random values
            frequency: frequency vector corresponding to stream vector
    '''
    np.random.seed(seed)
    p = min(1, k*q / np.sqrt(n))
    stream = np.zeros(m, dtype=np.int32)
    for i in range(m):
        if np.random.binomial(1, p):
            stream[i] = np.random.randint(1, q+1)
        else:
            stream[i] = np.random.randint(q+1, n+1)
    frequency = np.zeros(n)
    for s_i in stream:
        frequency[s_i-1] = frequency[s_i-1] + 1
    return stream, frequency

def plot_stream(stream):
    '''
    Description:
        Creates a histogram of the generated stream
    Inputs:
        stream: stream of random values
        n_bins: number of histogram 
    Outputs:
        histogram of stream values
    '''
    _, ax = plt.subplots()
    vals, freqs = np.unique(stream, return_counts=True)
    ax.bar(vals, freqs)
    ax.set_xlabel("Stream Item")
    ax.set_ylabel("Frequency")
    ax.set_title("Stream Item Frequency")
    plt.show()
