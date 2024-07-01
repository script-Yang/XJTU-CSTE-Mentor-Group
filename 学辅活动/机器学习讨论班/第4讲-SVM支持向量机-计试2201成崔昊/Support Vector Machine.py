import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class SVM:
    """
    默认多分类方法是one versus rest
    C:控制惩罚程度，越大则偏离界面的惩罚越多
    gamma：控制高斯核函数的系数，gamma越大越容易过拟合
    kernel：核方法
    """

    def __init__(self, C=1., kernel='rbf', gamma=0.1, ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.data = None
        self.Gram = None
        self.label = None
        self.num_class = None
        self.num_feature = None
        self.num_sample = None
        self.fitted = False
        self.weight = None
        assert kernel in ['linear', 'rbf']

    def coef(self):
        return (self.label*self.weight[:,:-1] @ self.data, self.weight[:,-1])

    def support_vector(self):
        return [np.where(self.weight[i] != 0) for i in range(self.num_class)]

    def fit(self, data, label):
        self._check_data(data, label)
        self._count_class()
        self.Gram = self._Gram(self.data)
        self.weight = np.zeros((self.num_class, self.num_sample + 1))
        self.label = np.zeros((self.num_class,self.num_sample))
        for clas in range(self.num_class):
            self.label[clas] = label
            self.label[clas][self.label[clas] != clas] = -1
            self.label[clas][self.label[clas] == clas] = 1
        for clas in range(self.num_class):
            self._SMO(self.label[clas], self.weight[clas])
        self.fitted = True

    def _check_data(self, data, label=np.zeros(1), train=True):
        if train:
            self.num_sample, self.num_feature = data.shape
            if label.any():
                assert self.num_sample == len(label)
            self.data = data
            self.label = label
        else:
            assert data.shape[1] == self.data.shape[1]
        return data, label

    def _count_class(self):
        self.num_class = len(np.unique(self.label))

    def _Gram(self,matrix):
        Gram = np.zeros((self.num_sample, len(matrix)))
        if self.kernel == 'linear':
            Gram = self.data @ matrix.T
        elif self.kernel == 'rbf':
            for row in range(self.num_sample):
                Gram[row, :] = np.exp(-self.gamma * np.sum((matrix - self.data[row]) ** 2, axis=1))
        return Gram

    def _SMO(self, label, weight, epsilon=1e-5, max_iter=100):
        iter = 0
        for _ in range(max_iter):
            severe = 0
            a1 = None
            res = (label * weight[:-1].reshape(1,-1) @ self.Gram + weight[-1]).reshape(-1)
            loss = res - label
            for i in range(self.num_sample):
                if 0 < weight[i] < self.C:
                    if severe < abs(1 - res[i] * label[i]):
                        severe = abs(1 - res[i]*label[i])
                        a1 = i
            if severe < epsilon:
                for i in range(self.num_sample):
                    if weight[i] == 0 and res[i]*label[i] < 1:
                        if severe < 1 - res[i]*label[i]:
                            severe = 1 - res[i]*label[i]
                            a1 = i
                    elif weight[i] == self.C and res[i]*label[i] > 1:
                        if severe < res[i]*label[i] - 1:
                            severe = res[i] * label[i] - 1
                            a1 = i
                    elif res[i]*label[i] != 1:
                        if severe < abs(1 - res[i]*label[i]):
                            severe = abs(1 - res[i]*label[i])
                            a1 = i
            if severe < epsilon:
                print('out')
                break
            a2 = a1
            while a2 == a1 or eta < epsilon:
                a2 = np.random.randint(self.num_sample)
                eta = self.Gram[a1, a1] + self.Gram[a2, a2] - 2 * self.Gram[a1, a2]
            if label[a1] != label[a2]:
                L = max(0., weight[a2] - weight[a1])
                H = min(self.C, self.C + weight[a2] - weight[a1])
            else:
                L = max(0., weight[a2] + weight[a1] - self.C)
                H = min(self.C, weight[a2] + weight[a1])
            old_a1 = weight[a1]
            old_a2 = weight[a2]
            weight[a2] += label[a2] * (loss[a1] - loss[a2]) / eta
            weight[a2] = min(H,max(L,weight[a2]))
            weight[a1] += label[a1]*label[a2]*(old_a2 - weight[a2])
            new_b1 = -loss[a1] - label[a1]*self.Gram[a1,a1]*(weight[a1]-old_a1) - label[a2]*self.Gram[a2,a1]*\
                              (weight[a2]-old_a2) + weight[-1]
            new_b2 = -loss[a2] - label[a1]*self.Gram[a2,a1]*(weight[a1]-old_a1) - label[a2]*self.Gram[a2,a2]*\
                              (weight[a2]-old_a2) + weight[-1]
            if abs(weight[a1]) < epsilon:
                weight[a1] = 0
            elif self.C - epsilon < weight[a1] < self.C + epsilon:
                weight[a1] = self.C
            if abs(weight[a2]) < epsilon:
                weight[a2] = 0
            elif self.C - epsilon < weight[a2] < self.C + epsilon:
                weight[a2] = self.C
            if 0 < weight[a1] < self.C:
                weight[-1] = new_b1
            elif 0 < weight[a2] < self.C:
                weight[-1] = new_b2
            else:
                weight[-1] = (new_b1+new_b2)/2
            if (abs(weight[a1] - old_a1) + abs(weight[a2]-old_a2) > epsilon):
                iter += 1
                if iter == max_iter:
                    return

    def predict(self, test_data):
        assert self.fitted
        test_x, _ = self._check_data(test_data, train=False)
        Gram = self._Gram(test_x)
        num_test = len(test_x)
        scores = self.label*self.weight[:,:-1] @ Gram + self.weight[:,-1].reshape(-1,1) # num_class x num_test
        preds = np.argmax(scores,axis=0)
        return preds

    def score(self, test_data, test_label):
        pred_y = self.predict(test_data)
        confusion_m = np.zeros((self.num_class, self.num_class))
        for true_l in range(self.num_class):
            for pred_l in range(self.num_class):
                confusion_m[true_l, pred_l] = np.sum([pred_y[test_label == true_l] == pred_l])
        print("confusion matrix:\n", confusion_m)
        trues = confusion_m.trace()
        print("accuracy:%f%%" % (100 * trues / len(test_label)))

if __name__ == "__main__":
    # digit = load_digits()
    # X, y = digit.data, digit.target
    # pca = PCA(n_components=16)
    # pca.fit(X)
    # X = pca.transform(X)
    # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=0)
    # clf = SVM(kernel='rbf',C=1,gamma=0.001)
    # clf.fit(train_x,train_y)
    # clf.score(test_x,test_y)
    iris = load_iris()
    X, y = iris.data, iris.target
    X,y = X[:,2:],y
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=0)
    clf = SVM(kernel='linear',C=1,gamma=0.1)
    clf.fit(train_x,train_y)
    clf.score(test_x,test_y)
    plt.scatter(X[:50,0],X[:50,1],c='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], c='purple')
    plt.scatter(X[100:, 0], X[100:, 1], c='green')
    w,b = clf.coef()
    print(w,b)
    x = np.linspace(1,4,100)
    for i in range(3):
        y = -w[i,0]/w[i,1]*x - b[i]/w[i,1]
        plt.plot(x,y)
    sv = clf.support_vector()
    print(sv)
    plt.scatter(X[sv[0],0],X[sv[0],1],c='red',s=5)
    plt.show()

