import numpy as np
from cvxopt import matrix,solvers, mul, spmatrix

x=np.array([[1.,6.],[1.,8.],[4.,11.],[5.,2.],[7.,6.],[9.,3.]])
xt=np.transpose(x)
XXt=np.dot(x,xt)
y=np.array([[1.],[1.],[1.],[-1.],[-1.],[-1.]])
yt=np.transpose(y)
yyt=np.dot(y,yt)
H=np.multiply(XXt,yyt)
H=matrix(H)

f=matrix([-1.,-1.,-1.,-1.,-1.,-1.],(6,1),'d')
A=np.diag([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
A=matrix(A)
a=matrix([0.,0.,0.,0.,0.,0.],(6,1),'d')
B=matrix([1,1,1,-1,-1,-1],(1,6),'d')
b=matrix(0.0,(1,1),'d')

sol=solvers.qp(H,f,A,a,B,b)
print('\n','alpha_1=',sol['x'][0])
print('alpha_2=',sol['x'][1])
print('alpha_3=',sol['x'][2])
print('alpha_4=',sol['x'][3])
print('alpha_5=',sol['x'][4])
print('alpha_6=',sol['x'][5])

