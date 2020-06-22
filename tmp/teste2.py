# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:26:14 2020

@author: vivam
"""


#+++++++++++++++++++++++++++++++++
# Multi Layer Perceptron (MLP)
#+++++++++++++++++++++++++++++++++
# Softmax activation function and cross entropy
# cost function
# 1st layer: sigmoid
# 2nd layer: softmax
# cost function: cross entropy
def mlp(X, yd, L, maxiter = 1000, plot = True):
  # Input
  # X: X_train
  # yd: desired y (y_train)
  # L: number 1st layer neurons
  # maxiter: maximum number of iterations
  
  # Add column x0 = 1 for the bias
  X = np.insert(X, 0, 1, axis = 1)    
  
  # Transform yd to 1 of n classes
  yd = np.where(yd == -1, 0, yd)
  if yd.shape[1] == 1:
    yd = np.array([(yd[:,0] == 1)*1, (yd[:,0] == 0) * 1]).T
  
  # N: number of instances
  # m: number of features
  # nc: number of classes and 2nd layer neurons
  N, m = X.shape
  nc = yd.shape[1]

  # A and B are weight matrices of 1st and 2nd layers
  # Initialize A and B with random values
  A = np.random.rand(L, m)
  B = np.random.rand(nc, L)

  v = X @ A.T    # 1st layer input
  z = sigmoid(v) # 1st layer output
  u = z @ B.T    # 2nd layer input
  y = softmax(u, axis = 1) # 2nd layer output
  error = y - yd # Error is the difference between predicted and actual output
  SME = 1/N * np.sum(error ** 2)  # Squared mean error

  it = 0       # Iteration counter
  alpha = 0.5  # Learning rate
  vecSME = np.array([SME])  # Vector of SME
  while ((abs(SME) > 1e-5) & (it < maxiter)):
      it += 1
      # Compute direction of steepest decline
      dJdA, dJdB = MLP_derivative(X, yd, A, B)
      # Update learning rate
      alpha = calc_lr_mlp(X, yd, A, B, -dJdA, -dJdB)
      #alpha = calc_alfa(X, yd, A, B, -dJdA, -dJdB)
       # Update weight matrices
      A = A - alpha * dJdA 
      B = B - alpha * dJdB
      # Update MLP output
      v = X @ A.T
      z = sigmoid(v)
      u = z @ B.T
      y = softmax(u, axis = 1)
      # Measure the error
      error = y - yd
      SME = 1/N * np.sum(error ** 2)

      #print(EQM)
      vecSME = np.vstack([vecSME, SME])
  # Plot error vs iteration
  if plot:
    plt.plot(vecSME)
  return A, B


# Predict new values in MLP network
def fx_mlp(X, y, A, B):
  # Input
  # X, y: test sets
  # A, B: weight matrices
  # Output:
  # y_hat: predicted values
  # acc: predicted accuracy
  
  # Insert bias column
  X = np.insert(X, 0, 1, axis = 1)    
  # Convert y to 1 of n classification
  y = np.where(y == -1, 0, y_test)
  if y.shape[1] == 1:
    y = np.array([(y[:,0] == 1)*1, (y[:,0] == 0) * 1]).T

  v = X @ A.T     # 1st layer input
  z = sigmoid(v)  # 1st layer output
  u = z @ B.T     # 2nd layer input
  y_hat = softmax(u, axis = 1)  # 2nd layer output
  # Convert softmax output to 0 and 1
  y_hat = np.where(y_hat == np.max(y_hat, axis=1, 
                                   keepdims=True), 1, 0)
  # Compute the accuracy
  acc = np.mean(y_hat == y)
  
  return y_hat, acc


def MLP_derivative(X, yd, A, B):
  # Inputs
  # X, yd: 
  # A, B: weights matrices
  # Output
  # dJdA, dJdB: A and B derivatives
  
  # N: number of instances
  # m: number of features
  N, m = X.shape
  nc = yd.shape[1]  # Number of classes
  v = X @ A.T       # 1st layer input
  z = sigmoid(v)    # 1st layer output
  u = z @ B.T       # 2nd layer input
  y = softmax(u)    # 2nd layer output

  # Compute the derivatives
  dJdB = 1/N * ((y - yd).T @ z) 
  dJdA = 1/N * (((y - yd) @ B) * ((1 - z) * z)).T @ X
  
  return dJdA, dJdB

# Calculate the learning rate using bissection algorithm
# The learning rate is used in MLP for faster convergence
def calc_lr_mlp(X, yd, A, B, dirA, dirB):
  # Inputs
  # X, yd: MLP input and output matrices (train set)
  # A, B: MLP weights matrices
  # dirA, dirB: A and B direction of steepest decline
  # Output
  # lr_m: optimized learning rate

  np.random.seed(1234)
  epsilon = 1e-3
  hlmin = 1e-3
  lr_l = 0                # Lower lr
  lr_u = np.random.rand() # Upper lr

  # New A and B positions
  An = A + lr_u * dirA
  Bn = B + lr_u * dirB
  # Calculate the gradient of new position
  dJdA, dJdB = MLP_derivative(X = X, yd = yd, A = An, B = Bn)
  g = np.concatenate((dJdA.flatten('F'), dJdB.flatten('F')))
  d = np.concatenate((dirA.flatten('F'), dirB.flatten('F')))
  hl = g.T @ d

  while hl < 0:
      #
      lr_u *= 2
      # Calculate the new position
      An = A + lr_u * dirA
      Bn = B + lr_u * dirB
      # Calculate the gradient of new position
      dJdA, dJdB = MLP_derivative(X = X, yd = yd, A = An, B = Bn)
      g = np.concatenate((dJdA.flatten('F'), dJdB.flatten('F')))
      hl = g.T @ d

  # lr medium is the average of lrs
  lr_m = (lr_l + lr_u) / 2

  # Estimate the maximum number of iterations
  itmax = np.ceil(np.log ((lr_u - lr_l) / epsilon))

  it = 0  # Iteration counter
  while np.any(hl) > hlmin and it < itmax :
      An = A + lr_u * dirA
      Bn = B + lr_u * dirB
      # Calculate the gradient of new position
      dJdA, dJdB = MLP_derivative(X = X, yd = yd, A = An, B = Bn)

      g = np.concatenate((dJdA.flatten('F'), dJdB.flatten('F')))
      hl = g.T @ d

      if np.any(hl) > 0 :
          # Decrease upper lr
          lr_u = lr_m
      elif np.any(hl) < 0 :
          # Increase lower lr
          lr_l = lr_m
      else:
          break
      # lr medium is the lr average
      lr_m = (lr_l + lr_u) / 2
      # Increase number of iterations
      it += 1
  return lr_m

# Stable softmax function
def softmax(s, axis = 1):
  max_s = np.max(s, axis = axis, keepdims = True)
  e = np.exp(s - max_s)
  y =  e / np.sum(e, axis = axis, keepdims = True)
  return y


def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
  
def sigmoid(s):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (s >= 0)
    neg_mask = (s < 0)
    z = np.zeros_like(s)
    z[pos_mask] = np.exp(-s[pos_mask])
    z[neg_mask] = np.exp(s[neg_mask])
    top = np.ones_like(s)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
