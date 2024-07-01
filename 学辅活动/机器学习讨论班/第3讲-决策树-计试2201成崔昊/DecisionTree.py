import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class TreeNode:
    """
    feature：结点分裂选择特征,叶节点没有该项
    value：结点分裂特征的临界取值
    gini：该结点对应的基尼指数
    class_count: 每一类对应样本数（类数，）
    """

    def __init__(self, depth, gini, class_count):
        self.depth = depth
        self.gini = gini
        self.class_count = class_count
        self.num_sample = np.sum(self.class_count)
        self.class_ = np.argmax(self.class_count)
        self.left = None
        self.right = None
        self.father = None
        self.feature = None
        self.value = None

    def get_class(self):
        return self.class_

    def is_leaf(self):
        return not self.left and not self.right

    def is_lastbuttwo_layer(self):
        if self.is_leaf():
            return False
        elif self.left and not self.left.is_leaf():
            return False
        elif self.right and not self.right.is_leaf():
            return False
        return True

    def set_feature(self, feature, value):
        self.feature = feature
        self.value = value


class DecisionTree:
    """
    criterion: 评价准则 —— Gini为基尼不纯度， Entropy为信息增益比，不过这里只写了Gini，真正写起来才知道信增益太难算了
    max_depth: 决策树最大深度
    min_samples_leaf: 叶结点所需最小样本数
    min_samples_split: 结点再划分所需的最小样本数
    alpha: 决策树剪枝时使用的超参数，越大模型泛化越强
    augdata: data与label拼在一起形成的增广矩阵
    prior_print()：先序遍历，调试用
    bulid_tree()：递归建树，寻找最优划分特征与划分点
    search()：根据样本的值找到对应超矩阵
    purning()：剪枝
    _purning_loss()：剪枝时计算子树的损失函数
    """

    def __init__(self, data, label, **kwargs):
        self.criterion = kwargs.get("criterion", "Gini")
        self.max_depth = kwargs.get("max_depth", 50)
        self.min_samples_leaf = kwargs.get("min_samples_leaf", 1)
        self.min_samples_split = kwargs.get("min_samples_split", 2)
        self.alpha = kwargs.get("alpha", 0)
        self.data = data
        self.label = label
        self.root = None
        self.num_sample = data.shape[0]
        self.num_feature = data.shape[1]
        self.num_class = len(np.unique(self.label))
        self.bulid_tree()

    def bulid_tree(self):
        print("Buliding the tree...")
        gini, counter = self._gini(self.label)
        self.root = TreeNode(0, gini, counter)
        self._bulid_tree(np.concatenate((self.data, self.label.reshape(-1, 1)), axis=1), 1, self.root)
        if self.alpha > 0:
            self.purning()

    def _bulid_tree(self, augdata, depth, father):
        if (augdata.shape[0] < max(self.min_samples_split, 2 * self.min_samples_leaf) or father.gini == 0
                or depth == self.max_depth):
            return father
        feature_loss = np.ones((self.num_feature, 3))  # 第一列放特征的最小基尼不纯度，第二列放对应切分点, 第三列放索引
        for feature in range(self.num_feature):
            data = augdata[:, [feature, -1]]
            data.sort(axis=0)
            for i in range(self.min_samples_leaf - 1, data.shape[0] - self.min_samples_leaf):  # 左右各留最低叶节点个数的样本
                if data[i, 0] == data[i + 1, 0]:
                    continue
                splitter = (data[i, 0] + data[i + 1, 0]) / 2
                gini = self._evaluate_loss(data, i)
                if feature_loss[feature, 0] > gini:
                    feature_loss[feature, :] = [gini, splitter, i + 1]

        # 寻找最优划分特征，记录在父节点上,这里是否一定会有解暂时存疑
        best = np.argmin(feature_loss[:, 0])
        father.set_feature(best, feature_loss[best, 1])
        # 接下来划分数据，重复迭代建树
        augdata = augdata[np.argsort(augdata[:, best])]
        lgini, lcounter = self._gini(augdata[:int(feature_loss[best, 2]), -1])
        lnode = TreeNode(depth, lgini, lcounter)
        lnode.father = father
        father.left = self._bulid_tree(augdata[:int(feature_loss[best, 2])], depth + 1, lnode)
        rgini, rcounter = self._gini(augdata[int(feature_loss[best, 2]):, -1])
        rnode = TreeNode(depth, rgini, rcounter)
        rnode.father = father
        father.right = self._bulid_tree(augdata[int(feature_loss[best, 2]):], depth + 1, rnode)
        return father

    def _evaluate_loss(self, augdata, split_pos):
        prop = (split_pos + 1) / augdata.shape[0]
        res = 1
        counter = np.bincount(augdata[:split_pos,1].astype(np.int64),minlength=self.num_class)
        res -= np.sum(np.square(counter / (split_pos + 1))) * prop
        gini, _ = self._gini(augdata[:split_pos + 1, 1])
        res += (gini - 1) * prop
        gini2, _ = self._gini(augdata[split_pos + 1:, 1])
        res += (gini2 - 1) * (1 - prop)
        return res

    def _gini(self, label):
        gini = 1
        if label.dtype != 'int64':
            label = label.astype(np.int64)
        counter = np.bincount(label,minlength=self.num_class)
        gini -= np.sum(np.square(counter / len(label)))
        return gini, counter

    def search(self, sample):
        now = self.root
        while now:
            if now.is_leaf():
                return now.get_class()
            elif sample[now.feature] <= now.value:
                now = now.left
            else:
                now = now.right

    def prior_print(self):
        self._prior_print(self.root)

    def _prior_print(self, node):
        if node.is_leaf():
            print(node.class_count, node.gini)
        else:
            print(node.feature, node.value, node.gini)
            self._prior_print(node.left)
            self._prior_print(node.right)

    def purning(self):
        print("Purning the tree...")
        while True:
            gt, cut_node = self._purning(self.root)
            if gt < self.alpha:
                cut_node.left = cut_node.right = None
                cut_node.feature = cut_node.value = None
            else:
                break
            if cut_node == self.root:
                break

    def _purning(self, node):
        """
        后序遍历自下而上计算
        这里可以优化，把左右子树的gini传到父结点，减少重复计算，不过这会使函数返回更多参数，鉴于这只是演示代码，故不这样降低可读性
        """
        minn, minn_node = 0xffffff, None
        if not node.left.is_leaf():
            res, res_node = self._purning(node.left)
            if res < minn:
                minn, minn_node = res, res_node
        if not node.right.is_leaf():
            res, res_node = self._purning(node.right)
            if res < minn:
                minn, minn_node = res, res_node
        if node.is_lastbuttwo_layer():
            subtree_num, subtree_loss = self._purning_loss(node)
            res = (node.gini * node.num_sample - subtree_loss) / (subtree_num - 1)
        else:
            res = 0xffffff
        if res < minn:
            minn, minn_node = res, node
        return minn, minn_node

    def _purning_loss(self, node):
        if node.is_leaf():
            return 1, node.gini * node.num_sample
        else:
            lnum, lloss = self._purning_loss(node.left)
            rnum, rloss = self._purning_loss(node.right)
            return 1 + lnum + rnum, lloss + rloss


class DTClassifier:
    """
    CART分类树
    """
    def __init__(self, max_depth=50, min_samples_leaf=1, min_samples_split=2, criterion="Gini", alpha=0.):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.alpha = alpha
        self.data = None
        self.label = None
        self.num_class = None
        self.num_feature = None
        self.num_sample = None
        self.fitted = False
        self.Tree = None

    def fit(self, data, label):
        self._check_data(data, label)
        self._count_class()
        self.Tree = DecisionTree(data, label, criterion=self.criterion, max_depth=self.max_depth, alpha=self.alpha,
                                 min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split)
        self.fitted = True

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

    def _count_class(self):
        self.num_class = len(np.unique(self.label))

    def predict(self, test_data):
        assert self.fitted
        test_x, _ = self._check_data(test_data, train=False)
        num_test = len(test_x)
        preds = np.zeros(num_test)
        for index, sample in enumerate(test_x):
            preds[index] = self.Tree.search(sample)
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
    digit = load_iris()
    X, y = digit.data, digit.target
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=0)
    clf = DTClassifier(max_depth=12, min_samples_leaf=5, alpha=0.4)
    clf.fit(train_x, train_y)
    clf.score(test_x, test_y)
    clf2 = DecisionTreeClassifier()
    clf2.fit(train_x, train_y)
    print(clf2.score(test_x, test_y))

