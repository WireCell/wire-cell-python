'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)


xdata=[]
ydata=[]
zdata=[]

import json
# read file
with open('depo/0/0-test.json', 'r') as myfile:
	data=myfile.read()
# parse file
obj = json.loads(data)
for depo in obj['depos']:
	if depo == {}: break # EOS
	x = depo['x']
	y = depo['y']
	z = depo['z']
	if x>-3100 and x<-1800 and y>2800 and y<6000 and z>3000 and z<4500:
		xdata.append(depo['z'])
		ydata.append(depo['x'])
		zdata.append(depo['y'])

xdata = np.array(xdata)
ydata = np.array(ydata)
zdata = np.array(zdata)

ax.scatter(xdata, ydata, zdata, s=0.01) #, c=zdata, cmap='Greens');
ax.set_xlabel('Beam Direction')
ax.set_ylabel('Drift Direction')
ax.set_zlabel('Height')

plt.show()

