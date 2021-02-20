import numpy as np

''' Defining the Linear regression class
'''

class LinearRegression:
    ''' __init__ method with default learning rate and number of iters;
        void weights and bias
    '''
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    ''' fit method that takes X training samples and y corresponding labels
        it uses the training step and the gradient descent
    '''
    def fit(self, X, y):
        # init parameters
        
        # in this case, (80,1) means 80 rows(samples) with 1 column each
        # (feature)
        n_samples, n_features = X.shape
        
        # for each component we put in a zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        #iterative gradient descent
        for _ in range(self.n_iters):

            # multiplies X w/ the weights
            y_predicted = np.dot(X, self.weights) + self.bias

            #dw (derivative with respect to w)
            #sum product is the same with the dot product
            #it is along the other axis compared to y_predicted 
            dw = (1/n_samples) * np.dot (X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            #update the weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            

    ''' predict method; takes new test samples and aproximates and returns
        the value
    '''
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
