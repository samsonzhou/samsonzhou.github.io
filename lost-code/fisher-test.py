import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import statistics

#number of coordinates in frequency vector
numPoints = 500
#number of buckets in CountSketch
numBuckets = 25
#number of repetitions in experiments
numExpers = 50000

items = []
for i in range(numPoints):
    items.append(numPoints**2/(i+1))
items = np.array(items)

allFreqs = []

#run experiments
for step in range(numExpers):
    freq = 0
    #hash coordinates
    selected = (np.random.rand(numPoints) < 1/numBuckets)
    #give signs to coordinates
    signs = (np.random.rand(numPoints) > .5)*2 - 1
    #fix first coordinate
    #0 for noise, 1 for estimate
    selected[0] = 0
    signs[0] = 1
    #signed sum in CountSketch bucket
    freq = np.sum(items * selected * signs)
    allFreqs.append(freq)

plt.hist(allFreqs,bins=100)

mean = np.mean(allFreqs)
variance = np.var(allFreqs)
sigma = np.sqrt(variance)
x = np.linspace(min(allFreqs), max(allFreqs), 100)
#plt.plot(x, np.exp(x, mean, sigma))
plt.plot(x, np.exp(-np.abs(x) / (.6*np.std(allFreqs)))*5000)
plt.show()


##mu, std = norm.fit(allFreqs)
##xmin, xmax = plt.xlim()
##x = np.linspace(xmin, xmax, 100)
##p = norm.pdf(x, mu, std)
##
##plt.plot(x, p, 'k', linewidth=10)
##title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
##plt.title(title)
##plt.show()


                
