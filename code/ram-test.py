import numpy as np
from utils import hash_func
def count_sketch(stream, r, n, b, eps):
    '''
    Description:
        implementation of count-sketch algorithm to determine heavy-hitters
    Parameters:
        stream: stream vector of random values
        r: number of algorithm iterations
        n: size of frequency vector
        b: number of buckets (hyperparam)
        eps: desired accuracy (hyperparam)
    Outputs:
        returns list of heavy hitter stream values
    '''
    heavy_hitters = []
    buckets_list = np.zeros((r, b))
    approx_norms = np.zeros(r)
    # process the stream
    for s_i in stream:
        for i in range(r):
            curr_buckets = buckets_list[i]
            v_i = 2*hash_func(i+100, s_i, 2) - 1
            bucket = hash_func(i, s_i, b)
            curr_buckets[bucket] += v_i
            approx_norms[i] += v_i

    # determine the threshold
    thresh = eps * np.sqrt(np.median(approx_norms**2))
    # print(buckets_list)
    for j in range(1, n+1):
        f_j = []
        for i in range(r):
            sign = 2*hash_func(i+100, j, 2) - 1
            bucket = hash_func(i, j, b)
            f_j.append(buckets_list[i][bucket] * sign)
        est_norm = np.median(f_j)
        if (est_norm > thresh):
                heavy_hitters.append(j)
    return heavy_hitters



from utils import generate_stream_skewed, plot_stream

n=100
skwd_strm, skwd_freq = generate_stream_skewed(n=n, m=int(1e6), q=1, k=2)
#plot_stream(skwd_strm)
hh=count_sketch(stream=skwd_strm[:int(1e5)], r=5, n=n, b=300, eps=0.1)
#print("Heavy Hitters from CS:")
#hh

import psutil, os

process = psutil.Process(os.getpid())
mem_info = process.memory_info()

print("RSS (resident memory):", mem_info.rss / 1024**2, "MB")
print("VMS (virtual memory):", mem_info.vms / 1024**2, "MB")
