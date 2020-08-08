import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class LRGD(object):
    """
        Linear Logistic Regression Classifier with batch steepest descent method
        
        Parameters
        -----------------
        eta: float 
            Learning rate.
        n_iter: int
            Passes over the dataset.
    """
    def __init__(self, eta=0.05, n_iter=100, lamda=0, random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.lamda = lamda
        self.random_state = random_state

    def prob(self, x, y):
        return 1 / (1 + np.exp(- y * self.net_input(x)))

    def net_input(self, x):
        return x.dot(self.w_)

    def fit(self, x, y):
        n_classes = len(np.unique(y))

        rgen = np.random.RandomState(self.random_state)
        if n_classes == 2:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], 1))
        else:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], n_classes))

        self.cost_ = []

        for i in range(self.n_iter):
            p = self.prob(x, y)

            regularization = self.lamda * self.w_.T.dot(self.w_)
            regularization = regularization.ravel()
            J = np.sum(-np.log(p) + regularization)

            grad = np.sum(-y*x*(1 - p), axis=0, keepdims=True).T + 2 * self.lamda * self.w_
            
            self.w_ -= self.eta * grad
            
            self.cost_.append(J)
        return self
    def predict(self, x):
        return 2*(self.net_input(x) > 0) - 1
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)

class MultiClassLRGD(object):
    """
        Linear Logistic Regression Classifier with batch steepest descent method
        
        Parameters
        -----------------
        eta: float 
            Learning rate.
        n_iter: int
            Passes over the dataset.
    """
    def __init__(self, eta=0.05, n_iter=100, lamda=0, random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.lamda = lamda
        self.random_state = random_state

    def net_input(self, x):
        return x.dot(self.w_)

    def softmax(self, z):
        # log_c = -np.max(z, axis=1)
        # log_c = log_c.reshape(-1, 1)
        # prob = np.exp(z + log_c)
        # prob = prob / np.exp(z + log_c).sum(axis=1).reshape(-1, 1)
        # return np.clip(prob, 1e-15, 1-1e-15)
        return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)

    def one_hot_encoder(self, y, n_classes):
        y_ohe = np.zeros((len(y), n_classes))
        for idx, i in enumerate(y):
            y_ohe[idx, int(i)] = 1
        return y_ohe

    def compute_cost(self, X, y):
        n_classes = len(np.unique(y))
        y_pred_enc = self.one_hot_encoder(y, n_classes)
        z = self.net_input(X)
        activation = self.softmax(z)
        cross_entropy = - np.sum(np.log(activation) * (y_pred_enc), axis = 1)
        regularization = self.lamda * np.sum(self.w_**2) / 2
        cross_entropy = cross_entropy + regularization
        return np.mean(cross_entropy)
        
    def fit(self, x, y):
        n_classes = len(np.unique(y))
        y_ohe = self.one_hot_encoder(y, n_classes)

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], n_classes))
    
        self.cost_ = []

        for i in range(self.n_iter):
            z = self.net_input(x)
            # print(z.shape)
            activation = self.softmax(z)
            # print(activation.shape)
            diff = activation - y_ohe
            grad = x.T.dot(diff)
            J = self.compute_cost(x, y)
            self.cost_.append(J)

            self.w_ -= self.eta * (grad + self.lamda * self.w_)
            # self.w_[0] -= self.eta * diff.sum(axis = 0)
            
        return self
    # def predict(self, x):
    #     return 2*(self.net_input(x) > 0) - 1
    def predict(self, x):
        z = self.net_input(x)
        activation = self.softmax(z)
        y_predicted = activation.argmax(axis=1)
        return y_predicted
    def score(self, x, y):
        return np.mean(self.predict(x) == y)

class MultiClassNewtonGD(object):
    """
        Linear Logistic Regression Classifier with batch steepest descent method
        
        Parameters
        -----------------
        eta: float 
            Learning rate.
        n_iter: int
            Passes over the dataset.
    """
    def __init__(self, eta=0.05, n_iter=100, lamda=0, random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.lamda = lamda
        self.random_state = random_state

    def net_input(self, x):
        return x.dot(self.w_)

    def softmax(self, z):
        # log_c = -np.max(z, axis=1)
        # log_c = log_c.reshape(-1, 1)
        # prob = np.exp(z + log_c)
        # prob = prob / np.exp(z + log_c).sum(axis=1).reshape(-1, 1)
        # return np.clip(prob, 1e-15, 1-1e-15)
        return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)

    def one_hot_encoder(self, y, n_classes):
        y_ohe = np.zeros((len(y), n_classes))
        for idx, i in enumerate(y):
            y_ohe[idx, int(i)] = 1
        return y_ohe

    def compute_cost(self, X, y):
        n_classes = len(np.unique(y))
        y_pred_enc = self.one_hot_encoder(y, n_classes)
        z = self.net_input(X)
        activation = self.softmax(z)
        cross_entropy = - np.sum(np.log(activation) * (y_pred_enc), axis = 1)
        regularization = self.lamda * np.sum(self.w_**2) / 2
        cross_entropy = cross_entropy + regularization
        return np.mean(cross_entropy)
        
    def fit(self, x, y):
        n_classes = len(np.unique(y))
        d = x.shape[1]
        dk = d * n_classes

        y_ohe = self.one_hot_encoder(y, n_classes)

        HT = np.zeros((d, n_classes, d, n_classes))
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], n_classes))
        w_shape = self.w_.shape
        self.cost_ = []

        for i in range(self.n_iter):
            z = self.net_input(x)

            activation = self.softmax(z)

            for j in range(n_classes):
                for k in range(n_classes):
                    r = np.multiply(activation[:,j],((j==k)-activation[:,k]))
                    HT[:,j,:,k] = np.dot(np.multiply(x.T,r),x)
            diff = activation - y_ohe
            grad = x.T.dot(diff)
            H = np.reshape(HT, (dk, dk))

            J = self.compute_cost(x, y)
            self.cost_.append(J)

            temp = self.w_.reshape(-1) - np.dot(np.linalg.pinv(H), grad.reshape(-1))
            self.w_ = temp.reshape(self.w_.shape)
            
        return self
    def predict(self, x):
        z = self.net_input(x)
        activation = self.softmax(z)
        y_predicted = activation.argmax(axis = 1)
        return y_predicted
    def score(self, x, y):
        return np.mean(self.predict(x) == y)

class LRNewton(object):
    """
        Linear Logistic Regression Classifier with batch steepest descent method
        
        Parameters
        -----------------
        eta: float 
            Learning rate.
        n_iter: int
            Passes over the dataset.
    """
    def __init__(self, eta=0.05, n_iter=100, lamda=0, random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.lamda = lamda
        self.random_state = random_state
        
    def prob(self, x, y):
        return 1 / (1 + np.exp(- y * self.net_input(x)))

    def net_input(self, x):
        return x.dot(self.w_)

    def fit(self, x, y):
        n_classes = len(np.unique(y))
        rgen = np.random.RandomState(self.random_state)
        if n_classes == 2:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], 1))
        else:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], n_classes))
        self.cost_ = []
        self.w_hist_ = []
        
        for i in range(self.n_iter):
            z = self.net_input(x)
            p = self.prob(x, y)
            
            regularization = self.lamda * self.w_.T.dot(self.w_)
            regularization = regularization.ravel()
            J = np.sum(-np.log(p) + regularization)
            
            grad = np.sum(-y*x*(1 - p), axis=0, keepdims=True).T + 2 * self.lamda * self.w_
            
            p_grad = np.exp(-y*z)*y*x / p / p
            second_order_grad = np.sum(y*x*p_grad, axis=0, keepdims=True).T + 2 * self.lamda
            
            self.w_ -= self.eta * grad / (second_order_grad + 1e-7)
  
            self.cost_.append(J)

            self.w_hist_.append(self.eta * grad / (second_order_grad + 1e-7))
        return self
    def predict(self, x):
        return 2*(self.net_input(x) > 0) - 1
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)

# if __name__=="__main__":
#     n = 200
#     x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
#     W = np.array([[ 2,  -1, 0.5,],
#                 [-3,   2,   1,],
#                 [ 1,   2,   3]])
#     y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
#                             + 0.5 * np.random.randn(n, 3), axis=1)
#     y_d5 = y_d5.reshape(-1, 1)

#     from sklearn import datasets
#     from sklearn.preprocessing import StandardScaler
#     # wine = datasets.load_wine()
#     # x_d5 = wine.data
#     # y_d5 = wine.target
#     # stdsc = StandardScaler()
#     # x_d5 = stdsc.fit_transform(x_d5)

#     # iris = datasets.load_iris()
#     # x_d5 = iris.data
#     # y_d5 = iris.target

#     # stdsc = StandardScaler()
#     # x_d5 = stdsc.fit_transform(x_d5)

#     print(y_d5.shape)
#     print(int(y_d5.max() + 1))
#     random_seed = 0
#     n_iter = 5
#     eta = 0.01
#     lamda = 0.0001

#     clf_gd = MultiClassNewtonGD(n_iter=n_iter, eta=eta, lamda=lamda, random_state=random_seed)
#     clf_gd.fit(x_d5, y_d5)
#     y_pred = clf_gd.predict(x_d5)
#     print("Accuracy:", clf_gd.score(x_d5, y_d5))
#     print("Weights: ", clf_gd.w_)
#     plt.plot(range(n_iter), clf_gd.cost_)
#     plt.xlabel("Iterations")
#     plt.ylabel("J(w)")
#     plt.show()