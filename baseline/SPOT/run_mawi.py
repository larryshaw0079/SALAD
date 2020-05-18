import numpy as np
import matplotlib.pyplot as plt
from spot import SPOT
import pandas as pd

f17 = './mawi_170812_50_50.csv'
f18 = './mawi_180812_50_50.csv'

P17 = pd.read_csv(f17)
P18 = pd.read_csv(f18)

X17 = P17['rSYN'].values
X18 = P18['rSYN'].values

n_init = 1000
init_data = X17[-n_init:] 	# initial batch
data = X18 	   		# stream

q = 1e-4 			# risk parameter
s = SPOT(q) 		# SPOT object
s.fit(init_data,data) 	# data import
s.initialize() 		# initialization step
results = s.run() 	# run
fig = s.plot(results) 	# plot
# plt.show(fig)
