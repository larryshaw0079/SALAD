import numpy as np
import matplotlib.pyplot as plt
from spot import bidSPOT,dSPOT
import pandas as pd

f = './edf_stocks.csv'

P = pd.read_csv(f)

# stream
u_data = (P['DATE'] == '2017-02-09')
data = P['LOW'][u_data].values

# initial batch
u_init_data = (P['DATE'] == '2017-02-08') | (P['DATE'] == '2017-02-07') | (P['DATE'] == '2017-02-06')
init_data = P['LOW'][u_init_data].values


q = 1e-5 			# risk parameter
d = 10				# depth
s = bidSPOT(q,d) 	# bidSPOT object
s.fit(init_data,data) 	# data import
s.initialize() 			# initialization step
results = s.run() 	# run
#del results['upper_thresholds'] # we can delete the upper thresholds
fig = s.plot(results) 		   # plot
# plt.plot(fig)
