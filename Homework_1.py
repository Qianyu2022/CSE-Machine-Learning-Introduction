'''
Homework 1 for CSE5819
@Author: Qiz19014
'''

# Exercise 1.2
from sympy.parsing.sympy_parser import parse_expr
from sympy import var, plot_implicit
'''
myplot = lambda exper:plot_implicit(parse_expr(exper))
expression='x**2+y**2-1'
myplot(expression)
'''

'''
from sympy import plot_implicit as pt, Eq
from sympy.abc import x,y
pt(Eq(x**2+y**2,1),line_color='r')
'''

from sympy import symbols
import math
from sympy import plot_implicit as pt, Eq
from sympy.plotting import plot, PlotGrid
#a = math.inf
x, y = symbols('x,y')
p1 = pt(Eq(x**2+y**2,1),line_color='r',show=False)
p2 = pt(Eq(abs(x)+abs(y),1),line_color='b',show=False)
p3 = pt(Eq(abs(x),1),(y,-1,1),line_color='g',show=False)
p4 = pt(Eq(abs(y),1),(x,-1,1),line_color='g',show=False)
#p5 = pt(Eq(abs(x)**a+abs(y)**a,1),line_color='g',show=True)
#p1.append(p2[0])
#p1.append(p3[0])

p1.extend(p2)
p1.extend(p3)
p1.extend(p4)
PlotGrid(1,1,p1,show=True,size=(6,6))


# Exercise 2

import numpy as np
import matplotlib.pyplot as plt

m = np.linspace(-5,5,100)
n = 1/(1+np.exp(-m))
p = 1/(1+np.exp(-2*m))
q = 1/(1+np.exp(-5*m))
plt.figure(figsize=(6,6))
plt.plot(m,n,'r')
plt.plot(m,p,'b')
plt.plot(m,q,'g')
plt.legend(['theta=1','theta=2','theta=5'])
plt.show()

# Exercise 3
import numpy as np
dat_file = r"q1x.dat"


with open(dat_file,'r') as file:
    Dat = []
    for line in file:
        if not line.strip() or line.startswith('@') or line.startswith('#'):
            continue
        row = line.split()
        Dat.append(float(row[0]))
        Dat.append(float(row[1]))

Dat = np.array(Dat)
Dat = Dat.reshape(99,2)
Const_feature = np.ones((99,1))
New_Dat = np.append(Const_feature,Dat,1)

dat_file_label = r"q1y.dat"
with open(dat_file_label,'r') as file_y:
    Label = []
    for line in file_y:
        if not line.strip() or line.startswith('@') or line.startswith('#'):
            continue
        row_y = line.split()
        Label.append(float(row_y[0]))


Label = np.array(Label)

Theta_initial= np.zeros((1,3))

'''
Gradient = np.zeros((1,3))
for i in range(99):
    Gradient = Gradient + (Label[i]-1/(1+math.exp(-np.dot(Theta_initial,np.transpose(New_Dat[i,:])))))
Norm_Gradient = abs(np.sqrt(Gradient[1,1]**2+Gradient[1,2]**2+Gradient[1,3]**2))


def logistic_regression(Theta_initial,alpha):
    theta = Theta_initial
    while Norm_Gradient > 0.000001:
        theta = theta-Gradient*alpha
        for j in range(99):
            Gradient = Gradient + (Label[i] - 1 / (1 + math.exp(-np.dot(Theta_initial, np.transpose(New_Dat[i, :])))))
        Norm_Gradient = abs(np.sqrt(Gradient[1, 1] ** 2 + Gradient[1, 2] ** 2 + Gradient[1, 3] ** 2))
'''

'''
def logistic_regression(iteration,alpha):
    theta = Theta_initial
    for j in range(iteration):
        Grad = np.zeros((1, 3))

        for i in range(99):
            Grad = Grad + New_Dat[i,:]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
        Gradient = Grad
        Norm_Gradient = abs(np.sqrt(Gradient[0, 0] ** 2 + Gradient[0, 1] ** 2 + Gradient[0, 2] ** 2))
        if Norm_Gradient <= 0.000001:
            break
        else:
            theta = theta - Gradient * alpha
        print(Gradient)


logistic_regression(0.01)
'''

def logistic_regression_count(alpha):
    theta = Theta_initial
    Grad_history = []
    Count = 0
    Norm_Gradient = 1
    while Norm_Gradient > 1e-6:

        Grad = np.zeros((1, 3))

        for i in range(99):
            Grad[0,0] = Grad[0,0] + New_Dat[i,0]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
            Grad[0,1] = Grad[0,1] + New_Dat[i,1]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
            Grad[0,2] = Grad[0,2] + New_Dat[i,2]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
        Gradient = Grad/99
        #print(Gradient)
        Grad_history.append(Gradient)
        Norm_Gradient = np.sqrt(Gradient[0, 0] ** 2 + Gradient[0, 1] ** 2 + Gradient[0, 2] ** 2)
        theta = theta + Gradient * alpha
        Count = Count + 1
        #print(theta)

    #Norm_Gradient = abs(np.sqrt(Gradient[0, 0] ** 2 + Gradient[0, 1] ** 2 + Gradient[0, 2] ** 2))
    print(Gradient)
    print(Count)
    return theta

Final_theta = logistic_regression_count(0.001)

def logistic_regression(iteration,alpha):
    theta = Theta_initial
    for j in range(iteration):

        Grad = np.zeros((1, 3))

        for i in range(99):
            Grad[0,0] = Grad[0,0] + New_Dat[i,0]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
            Grad[0,1] = Grad[0,1] + New_Dat[i,1]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
            Grad[0,2] = Grad[0,2] + New_Dat[i,2]*(Label[i] - 1 / (1 + np.exp(-np.dot(theta, np.transpose(New_Dat[i, :])))))
        Gradient = Grad/99
        #print(Gradient)

        theta = theta + Gradient * alpha
        #print(theta)

    Norm_Gradient = np.sqrt(Gradient[0, 0] ** 2 + Gradient[0, 1] ** 2 + Gradient[0, 2] ** 2)
    print(Norm_Gradient)
    return theta

Final_theta = logistic_regression(7000,0.1)

# Plot the data
import matplotlib.pyplot as plt
x_0 = Dat[0:49,0]
x_1 = Dat[50:98,0]
y_0 = Dat[0:49,1]
y_1 = Dat[50:98,1]

BoundCond_x = np.linspace(0,8,100)
BoundCond_y = -(Final_theta[0,1]/Final_theta[0,2])*BoundCond_x-(Final_theta[0,0]/Final_theta[0,2])
plt.scatter(x_0,y_0,c='r')
plt.scatter(x_1,y_1,c='b')
plt.plot(BoundCond_x,BoundCond_y,color='green')
plt.show()

# Prediction on new datapoints
Probability = 1/(1+np.exp(1 + np.exp(-np.dot(Final_theta, np.transpose([1,2,1])))))


#import pandas as pd
#df = pd.read_csv('q1x.dat',header=None)
