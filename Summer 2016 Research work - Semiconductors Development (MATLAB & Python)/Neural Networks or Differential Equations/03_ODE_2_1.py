#==============================================================================
""" Solving Second Order ODE using Artificial Neural Networks

d2y/dx2 = f(x,y,dy/dx), where:
    
    f(x,y,dy/dx) = -1/5*exp(-x/5)*cos(x) - 1/5*dy/dx - y
    
Boundary conditions: y(0) = 0 and d/dx(y=0) = 1

Input Layer size        = 1 (Single input of x space as 1-D array)
# Hidden Layers         = 1
# units in Hidden layer = 10
Output Layer size       = 1 (Single value output)
 
@author: Dhinesh"""
#==============================================================================

import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

#==============================================================================
# FUNCTION DEFINITIONS: 
#==============================================================================

# Analytical solution of the problem:  
def y_analytic_solution(x):

    return np.exp(-x/5.) * np.sin(x)

#d2y/dx2 = f(x,y,dy/dx)
def f(x, psy, dpsy):
    
    func = -1./5. * np.exp(-x/5.) * np.cos(x) - 1./5. * dpsy - psy
    
    return func

# Sigmoid function defined
def sigmoid(x):
    
    return 1. / (1. + np.exp(-x))

# Gradient of Sigmoid function
def sigmoid_grad(x):
    
    return sigmoid(x) * (1 - sigmoid(x))

# Neural Network Forward Propagation Output
def neural_network(W, x):
    
    a1 = x
    z2 = np.dot(x, W[0])
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W[1])
    
    return z3

def neural_network_x(x):
            
    a1 = x
    z2 = np.dot(x, W[0])
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W[1])
    
    return z3

    #a1 = sigmoid(np.dot(x, W[0]))
    #return np.dot(a1, W[1])

# Function reutrning trial solution of the form y(x) = x + x^2 * N(x,W)
def y_trial(xi, FF_net_out):
    
    return xi + xi**2 * FF_net_out


# Cost/loss function:
def loss_function(W, x):
    
    loss_sum = 0.
    
    for xi in x:
        
        FF_net_out = neural_network(W, xi)[0][0]

        d_dx_FF_net_out = grad(neural_network_x)(xi)
        
        d2x_dx2_FFnet_out = grad(grad(neural_network_x))(xi)
        
        y_t = y_trial(xi, FF_net_out)
        
        y_grad = grad(y_trial)
        
        y_grad2 = grad(y_grad)
        
        gradient_of_trial = y_grad(xi, FF_net_out)
        
        second_gradient_of_trial = y_grad2(xi, FF_net_out)
        
        func = f(xi, y_t, gradient_of_trial) # right part function
        
        err_sqr = (second_gradient_of_trial - func)**2
        
        loss_sum += err_sqr
        
    return loss_sum

#==============================================================================
# Neural Network Execution: 
#==============================================================================

nx = 20
dx = 1. / nx

x_space = np.linspace(0, 1, nx)    
y_analytic = y_analytic_solution(x_space)

W = [npr.randn(1, 10), npr.randn(10, 1)] #Random initialization of weights

lmb = 0.001 # Learning rate of the Neural Network


for i in range(100):
    
    loss_grad =  grad(loss_function)(W, x_space)

    W[0] = W[0] - lmb * loss_grad[0] # Upgrading weights
    W[1] = W[1] - lmb * loss_grad[1] # Upgrading weights
    
print("The minimized cost function is: {0:.4f} \n"
      .format(loss_function(W, x_space)))

y_nn = [y_trial(xi, neural_network(W, xi)[0][0]) for xi in x_space] 


print("Plot showing results:\n")
plt.figure()
plt.plot(x_space, y_analytic) 
#plt.plot(x_space, y_fd)
plt.plot(x_space, y_nn)
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.title('Solution of the First Order ODE', fontsize = 14)
plt.legend(['Analytic Solution','Neural Networks Solution'])
plt.show() 