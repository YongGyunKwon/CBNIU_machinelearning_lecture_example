import numpy as np
import cvxopt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class SVM:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.dot(X[i], X[j])
        H = cvxopt.matrix(np.outer(y,y)*K)
        f = cvxopt.matrix(np.ones(n_samples)*-1)
        B = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        A = cvxopt.matrix(np.diag(np.ones(n_samples)*-1))
        a = cvxopt.matrix(np.zeros(n_samples))
        solution = cvxopt.solvers.qp(H,f,A,a,B,b)
        
        a = np.ravel(solution['x'])
        sv = a > 1e-5
        
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /=len(self.a)
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w+=self.a[n] * self.sv_y[n]*self.sv[n]
    
    def project(self, X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        return np.sign(self.project(X))
    
X,y = make_blobs(n_samples=250, centers=2, random_state=0, cluster_std=0.60)
y[y==0] = -1
y = y.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svm=SVM()
svm.fit(X_train, y_train)

def f(x, w, b, c=0):
    return (-w[0]*x-b+c)/w[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')
a0 = -2; a1 = f(a0, svm.w, svm.b)
b0 = 4; b1 = f(b0, svm.w, svm.b)
plt.plot([a0,b0], [a1,b1], 'k')

a0 = -2; a1 = f(a0, svm.w, svm.b, 1)
b0 = 4; b1 = f(b0, svm.w, svm.b, 1)
plt.plot([a0,b0], [a1,b1], 'k--')

a0 = -2; a1 = f(a0, svm.w, svm.b, -1)
b0 = 4; b1 = f(b0, svm.w, svm.b, -1)
plt.plot([a0,b0], [a1,b1], 'k--')

y_pred = svm.predict(X_test)
print('training\n', confusion_matrix(y_test, y_pred))
y_ored = svm.predict(X_test)
print('test\n', confusion_matrix(y_test, y_pred))

plt.title('SVM')
plt.show()