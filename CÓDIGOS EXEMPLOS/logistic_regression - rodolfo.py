import numpy as np
import utils
from scipy.special import softmax


class logistic_regression:
    def logistic_function(self, s):
        return np.exp(s)/(1+np.exp(s))

    def map_y(self, prob_value, threshold=0.5):
        if prob_value > threshold:
            return 1
        else:
            return -1


    def fit(self, X, y, learning_rate=0.1):
        N= X.shape[0]
        dim = X.shape[1]

        iteration = 0
        sum_grad_error = 1000
        
        new_w = utils.initialize_w(dim)
        while sum_grad_error > 0.0001:
            w = new_w

            sum_equation_ = 0
            grad_error = 0
            

            for index, x in enumerate(X):
                equation_ = (y[index] * x)/(1 + np.exp(y[index] * w *x))
                sum_equation_ = sum_equation_ + equation_

            grad_error = -1/N * sum_equation_
            sum_grad_error = np.sum(np.absolute(grad_error))
            
            new_w = w - (learning_rate * grad_error)
            
            iteration+=1


        return new_w, grad_error, iteration


    def predict(self, X, w, threshold=0.5):
        y_pred = np.dot(X, w)
        probs = self.logistic_function(y_pred)
        y_pred = np.array([self.map_y(prob,threshold) for prob in probs])
        return y_pred


class multinomial_logistic_regression:
    
    def fit(self, X, y, k, learning_rate=0.1):
        N, d= X.shape
        
        iteration = 0
        sum_grad_error = 1000
        
        # W = d x k
        new_w = utils.initialize_w(dim = (d, k))
        
        while sum_grad_error > 0.1:
            w = new_w
            
            sum_equation_ = 0
            grad_error = 0
            
            for index, x in enumerate(X):
                #conforme indicado no artigo, arrange xi in X columnwise
                x_i = np.expand_dims(x, axis=1) # x_i.shape: d x 1 ->transpose xi
                y_i = np.expand_dims(y[index], axis=1) #  y_i.shape: k x 1
                
                #pi = f(W' * xi) = softmax(W' *xi) -> kxd * dx1 = shape k x 1
                pi = softmax(np.dot(w.T, x_i), axis=0) # shape k x 1 
                aux = pi - y_i # shape k x 1
                equation_ = -np.dot(x_i, aux.T) # dx1 * 1xk = dxk
                sum_equation_ = sum_equation_ + equation_
            
            
            grad_error = -1/N * sum_equation_
            sum_grad_error = np.sum(np.absolute(grad_error))

            new_w = w - (learning_rate * grad_error)
                    
            iteration+=1

            
        return new_w, grad_error, iteration


    def predict(self, X, w):
        return softmax(np.dot(w.T, X.T), axis=0).T # n x k
         