# SPOT (Streaming Peaks-Over-Threshold)

## INTRODUCTION
This folder contains a python3 file, a license file and several datasets on whch we can run SPOT and its variants.
The python file spot.py contains 4 classes : SPOT, DSPOT, biSPOT and biDSPOT.
The classical algorithms (SPOT and DSPOT) compute only upper threshold although
biSPOT and biDSPOT computes upper and lower thresholds.



## PYTHON3 DEPENDENCIES
The following packages are needed to run the algorithm

* `scipy`		[optimization]
* `numpy`		[array class]
* `pandas` 		[dataframe]
* `matplotlib` 	[plot]
* `tqdm`	[progress bar]

These libraries can be downloaded through pip3 :
`pip3 install scipy numpy pandas matplotlib tqdm --upgrade`




## DATASETS
The folder contains several datasets on which (D)SPOT can be run

* `mawi_170812_50_50.csv` and `mawi_180812_50_50.csv`
	These .csv gather aggregated NetFlow measures of MAWI network captures
	(17/08/2012 and 18/08/2012)

* `physics.dat`
	North-South Component of the magnetic field (in nT) from the ACE satellite
	between 1/1/1995 and 1/6/1995

* `edf_stocks.csv`
	EDF stocks prices from 23/01/2017 to 10/02/2017

* `rain.dat`
	Daily precipitation rate at (71.4262, -58.1250) from 1947 and 2007 in mm/s


## EXAMPLES
Here some examples are given to test the algorithms. We advice to use the ipython3 shell with matplotlib to run these script and display plots.

To install ipython3
	`sudo apt-get install ipython3`

To run from the command line
	`ipython3 --matplotlib`

Then you can copy/paste the scripts you want to run.



### Physics
```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from spot import bidSPOT

f = './physics.dat'
r = open(f,'r').read().split(',')
X = np.array(list(map(float,r)))

n_init = 2000
init_data = X[:n_init] 	# initial batch
data = X[n_init:]  		# stream

q = 1e-3 				# risk parameter
d = 450  				# depth parameter
s = bidSPOT(q,d)     	# biDSPOT object
s.fit(init_data,data) 	# data import
s.initialize() 	  		# initialization step
results = s.run()    	# run
s.plot(results) 	 	# plot
```



### Rain
```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from spot import SPOT

f = './rain.dat'
r = open(f,'r').read().split(',')
X = np.array(list(map(float,r)))

n_init = 1000
init_data = X[:n_init] 	# initial batch
data = X[n_init:] 		# stream

q = 1e-4  			# risk parameter
s = SPOT(q)  		# biDSPOT object
s.fit(init_data,data) 	# data import
s.initialize() 		# initialization step
results = s.run() 	# run
s.plot(results) 	# plot
```



### MAWI
```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from spot import SPOT
import pandas as pd

f17 = './mawi_170812_50_50.csv'
f18 = './mawi_180812_50_50.csv'

P17 = pd.DataFrame.from_csv(f17)
P18 = pd.DataFrame.from_csv(f18)

X17 = P17['rSYN'].values
X18 = P18['rSYN'].values

n_init = 1000
init_data = X17[-n_init:] 	# initial batch
data = X18 	   		# stream

q = 1e-4 			# risk parameter
s = SPOT(q) 		# SPOT object
s.fit(init_data,data) 	# data import
s.initialize() 		# initialization step
results = s.run() 	# run
s.plot(results) 	# plot
```


### EDF stocks
```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from spot import bidSPOT,dSPOT
import pandas as pd

f = './edf_stocks.csv'

P = pd.DataFrame.from_csv(f)

# stream
u_data = (P['DATE'] == '2017-02-09')
data = P['LOW'][u_data].values

# initial batch
u_init_data = (P['DATE'] == '2017-02-08') | (P['DATE'] == '2017-02-07') | (P['DATE'] == '2017-02-06')
init_data = P['LOW'][u_init_data].values


q = 1e-5 			# risk parameter
d = 10				# depth
s = bidSPOT(q,d) 	# bidSPOT object
s.fit(init_data,data) 	# data import
s.initialize() 			# initialization step
results = s.run() 	# run
#del results['upper_thresholds'] # we can delete the upper thresholds
s.plot(results) 		   # plot
```
