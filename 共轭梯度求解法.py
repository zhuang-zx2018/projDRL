# Author(s):    zhuan
# Date:         2025/3/7
# Time:         16:52

# Your code starts here
'''测试策略梯度函数'''
import numpy as np

A = np.array([[4,-2,-1],[-2,4,-2],[-1,-2,3]])
b = np.array([[0],[-2],[3]])

x = np.zeros((3,1))
r0 = np.dot(A,x)-b
p = -r0

for i in range(10):
    Ap = np.dot(A,p)
    alpha = np.dot(r0.T,r0)/np.dot(p.T,Ap)
    x = x+alpha*p
    r = r0+alpha*Ap
    print(np.linalg.norm(r))
    if np.linalg.norm(r)< 0.001:
        break
    beta = np.dot(r.T,r)/np.dot(r0.T,r0)
    p = -r+beta*p
    r0 = r
print('迭代了{:d}步'.format(i))
print(np.dot(A,x)-b)#最终结果和实际结果的差
