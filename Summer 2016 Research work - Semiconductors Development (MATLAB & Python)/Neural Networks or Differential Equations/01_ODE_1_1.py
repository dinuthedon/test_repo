#==============================================================================
""" Solving First Order ODE using Artificial Neural Networks

dy/dx + A(x) * y = B(x), where:
    
    A(x) = (x+(1+3x^2)/(1+x+x^3))
    
    B(x) = x^3+2x+x^2+((1+3x^2)/(1+x+x^3))
    
We write the ODE as:
    
    dy/dx = f(x,y) = B(x) - A(x) * y

Input Layer size        = 1 (Single input of x space as 1-D array)
# Hidden Layers         = 1
# units in Hidden layer = 10
Output Layer size       = 1 (Single value output)
 
@author: Dhinesh"""
#==============================================================================

import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt

#==============================================================================
# FUNCTION DEFINITIONS: 
#==============================================================================

def A(x):

    return x + (1. + 3.*x**2) / (1. + x + x**3)

def B(x):

    return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))

#dy/dx = f(x,y)
def f(x,y):

    return B(x) - y * A(x)

# Analytical solution of the problem:    
def y_analytic_solution(x):

    return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2

def rand_initalize_weights(m,n):
    
    epsilon_init = 0.12
    W = npr.randn(n,m)*2*epsilon_init-epsilon_init
    
    return W
    

# Sigmoid function defined
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Gradient of Sigmoid function
def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


def neural_network(W, x):
    
    a1 = x
    z2 = np.dot(x, W[0])
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W[1])
    
    return z3


def d_FF_net_out_dxi(W, xi, k=1):
    
    prod_of_weights = np.dot(W[1].T, W[0].T**k)
    
    z_grad = sigmoid_grad(xi)
    
    dNet_dxi = np.dot(prod_of_weights, z_grad)
    
    return dNet_dxi


def loss_function(W, x):
    
    cost_function_sum = 0.
    
    for xi in x:
        
        FF_net_out = neural_network(W, xi)[0][0]
        
        y_trial = 1. + xi * FF_net_out
        
        d_net_out = d_FF_net_out_dxi(W, xi)[0][0]
        
        dy_dweight = FF_net_out + xi * d_net_out
        
        func = f(xi, y_trial)       
        
        err_sqr = (dy_dweight - func)**2

        cost_function_sum += err_sqr
        
    return cost_function_sum



#==============================================================================
# Neural Network Execution: 
#==============================================================================

nx = 15
dx = 1. / nx

x_space = np.linspace(0, 1, nx)    
y_analytic = y_analytic_solution(x_space)
y_fd = np.zeros_like(y_analytic)
y_fd[0] = 1. # Inital Condition

for i in range(1, len(x_space)):
    y_fd[i] = y_fd[i-1] + B(x_space[i]) * dx - y_fd[i-1] * A(x_space[i]) * dx
  
"""plt.figure()
plt.plot(x_space, y_analytic)
plt.plot(x_space, y_fd)
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.title('Solution of the First Order ODE', fontsize = 14)
plt.legend(['Analytic Solution','Finite Difference Solution'])
plt.show()"""

W = [npr.randn(1, 10), npr.randn(10, 1)] #Random initialization of weights

lmb = 0.001 # Learning rate of the Neural Network

for i in range(1000):

    #Gradient of loss funtion w.r.t. weights    
    loss_grad =  grad(loss_function)(W, x_space) 
    
    W[0] = W[0] - lmb * loss_grad[0] # Upgrading weights
    W[1] = W[1] - lmb * loss_grad[1] # Upgrading weights
    

print("The minimized cost function is: {0:.4f} \n"
      .format(loss_function(W, x_space)))

y_nn = [1 + xi * neural_network(W, xi)[0][0] for xi in x_space] 

print("Optimized weights for input layer mapping to hidden layer (Theta1): {}\n"
      .format(W[0]))
      
print("Optimized weights for hidden layer mapping to output (Theta2): {}\n"
      .format(W[1].T))

print("Plot showing results:\n")
plt.figure()
plt.plot(x_space, y_analytic) 
plt.plot(x_space, y_fd)
plt.plot(x_space, y_nn)
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.title('Solution of the First Order ODE', fontsize = 14)
plt.legend(['Analytic Solution','Finite Difference Solution', 
            'Neural Networks Solution'])
plt.show()