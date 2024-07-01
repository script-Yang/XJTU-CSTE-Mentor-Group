import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt


class Perceptron:
    """
    num_sample：样本个数
    num_feature：特征个数/维度
    num_class：分类类别数
    weight：模型参数，包含偏置
    misclassification：误分类的实例下标
    lr: learning_rate
    wd: weight_decay
    lr_decay: learning_rate_decay
    reg: regulazation type ('L1' or 'L2')
    reg_s: regulazation strength (only use for L1)
    """
    def __init__(self, lr, epochs, wd=1., lr_decay=1., reg='L2',reg_s=1):
        self.data = None
        self.label = None
        self.num_sample = None
        self.num_feature = None
        self.num_class = 2
        self.fitted = None
        self.weight = None
        self.misclassification = None
        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.lr_decay = lr_decay
        self.reg = reg
        self.reg_s = reg_s

    def fit(self, train_data, train_label):
        # 将偏置添加系数矩阵内
        aug_data = np.concatenate((train_data, np.ones((train_data.shape[0], 1))), axis=1)
        self.weight = np.random.normal(0, 0.1, (aug_data.shape[1],))
        self._check_data(aug_data, train_label)
        self.fitted = True
        for i in range(1, self.epochs+1):
            preds = self.data @ self.weight
            loss = self._sign_loss(preds)
            batch_size = min(len(self.misclassification), 5)
            batch = np.random.choice(self.misclassification, batch_size, replace=True)
            dw = self._gradient(batch)
            if self.reg == 'L1':
                self.weight -= self.lr*dw
            else:
                self.weight = self.weight*self.wd - self.lr*dw
            self.lr *= self.lr_decay
            if not i % 10:
                print(f"{i}/{self.epochs}  loss: ", loss)
        print("Done!")

    def predict(self, test_data):
        assert self.fitted
        aug_data = np.concatenate((test_data, np.ones((test_data.shape[0], 1))), axis=1)
        test_x, _ = self._check_data(aug_data, train=False)
        preds = test_x @ self.weight
        preds[preds >= 0] = 1
        preds[preds < 0] = -1
        return preds

    def score(self, test_data, test_label):
        pred_y = self.predict(test_data)
        confusion_m = np.zeros((self.num_class, self.num_class))
        for i, true_l in enumerate((-1, 1)):
            for j, pred_l in enumerate((-1, 1)):
                confusion_m[i, j] = np.sum([pred_y[test_label == true_l] == pred_l])

        print("confusion matrix:\n", confusion_m)
        trues = confusion_m.trace()
        print("accuracy:%f%%" % (100 * trues / len(test_label)))

    def _check_data(self, data, label=np.zeros(1), train=True):
        self.num_sample, self.num_feature = data.shape
        if label.any():
            assert self.num_sample == len(label)
        if train:
            self.data = data
            self.label = label
        else:
            assert data.shape[1] == self.data.shape[1]
        return data, label

    def _sign_loss(self, preds):
        loss = - self.label*preds
        loss[loss < 0] = 0
        self.misclassification = np.where(loss > 0)[0]
        if self.reg == 'L1':
            return np.sum(loss) + self.reg_s*np.sum(np.abs(self.weight[:-1]))
        else:
            return np.sum(loss)

    def _gradient(self, batch):
        X, y = self.data[batch], self.label[batch]
        grad = -np.sum(y.reshape(-1, 1)*X, axis=0)/len(batch)
        if self.reg == 'L1':
            np.dreg = np.ones(self.weight.shape)
            np.dreg[self.weight < 0] = -1
            np.dreg[-1] = 0
            np.dreg *= self.reg_s
            grad += np.dreg
        return grad


if __name__ == "__main__":
    digit = load_breast_cancer()
    X, y = digit.data, digit.target
    X = (X - np.mean(X, axis=0))/np.sqrt(np.var(X, axis=0))
    y[y == 0] = -1
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=0)
    clf = Perceptron(lr=10, epochs=200, wd = 0.99, lr_decay=1, reg='L2')
    clf.fit(train_x, train_y)
    clf.score(test_x, test_y)
    plt.plot(clf.weight)
    plt.show()
    #print(sorted(np.abs(clf.weight)))
    #clf2 = linear_model.Perceptron()
    #clf2.fit(train_x, train_y)
    #print(clf2.score(test_x, test_y))
