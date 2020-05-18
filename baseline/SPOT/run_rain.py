import numpy as np
import matplotlib.pyplot as plt
from spot import SPOT

f = './rain.dat'
r = open(f,'r').read().split(',')
X = np.array(list(map(float,r)))

n_init = 1000
init_data = X[:n_init] 	# initial batch
data = X[n_init:] 		# stream

q = 1e-4  			# risk parameter
s = SPOT(q)  		# biDSPOT object
s.fit(init_data,data) 	# data import
s.initialize() 		# initialization step
results = s.run() 	# run
fig = s.plot(results) 	# plot
# plt.plot(fig)
