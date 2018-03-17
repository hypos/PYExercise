import numpy as np
from NNetwork import NerualNetwork

xdata=[[0,0],[0,1],[1,0],[1,1]]
ydata=[0,1,1,0]

x=np.array(xdata)
y=np.array(ydata)
nn=NerualNetwork([2,2,1])
nn.fit(x,y)
for i in xdata:
    print(i,nn.predict(i))