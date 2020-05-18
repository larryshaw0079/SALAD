import numpy as np
import matplotlib.pyplot as plt
from spot import bidSPOT

f = './physics.dat'
r = open(f,'r').read().split(',')
X = np.array(list(map(float,r)))
print(X.shape)

n_init = 2000
init_data = X[:n_init] 	# initial batch
data = X[n_init:]  		# stream

q = 1e-3 				# risk parameter
d = 450  				# depth parameter
s = bidSPOT(q, d)     	# biDSPOT object
s.fit(init_data, data) 	# data import
s.initialize() 	  		# initialization step
results = s.run()    	# run
fig = s.plot(results) 	 	# plot
# plt.show(fig)
