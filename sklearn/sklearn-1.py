#import package
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

#1. define train dataset
X = [[0, 0], [1, 1]]
y = [0, 1]

#2. define train model
clf = svm.SVC()
clf.fit(X, y)

#3. use model to predict result
pred_res = clf.predict([[2, 2]])
print (pred_res)


#load data
iris = datasets.load_iris()
X = iris.data[:, :2] # reserve all rows, while only first 2 cols
y = iris.target # label

#train SVM model
C = 1.0
#svc = svm.SVC(kernel='linear', C=C).fit(X, y) # linear kernel
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y) # RBF kernel
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y) # multi-poly kernel
#lin_svc = svm.LinearSVC(C=C).fit(X, y) #linear SVM


# knn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborClassifier
iris = datasets.load_iris()
#导入数据和标签
iris_X = iris.data
iris_y = iris.target
#划分为训练集和测试集数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
#print(y_train)
#设置knn分类器
knn = KNeighborsClassifier()
#进行训练
knn.fit(X_train,y_train)
#使用训练好的knn进行数据预测
print(knn.predict(X_test))
print(y_test) 