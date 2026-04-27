"""
ref: https://www.jianshu.com/p/55aa444ea1cd
using FC NN to solve linear regression
"""

## 1. 数据模拟生成的数据集
# 生成用于拟合的模拟数据，本次采用线性的二元一次方程 
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # 随机种子，保证每次的数据的可重复行和一致性
from keras.models import Sequential
from keras.layers import Dense

# create some data
X = np.linspace(-1, 1, 1000) # 生成 -1到1的1000个随机数
np.random.shuffle(X)    # 打乱数据
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (1000, )) # Y数据随机扰动
# plot data
plt.scatter(X, Y)
plt.show()


# 数据集保存成csv格式文件
import pandas as pd  

save = pd.DataFrame({'Y_lbale':Y,'X_values':X})  
save.to_csv('mydata.csv',index=False,sep=',')

'''
data example:
X_values,Y_lbale
0.9559559559559558,2.4611472908040004
-0.96996996996997,1.5800108238723558
-0.8878878878878879,1.5441473485192092
0.6036036036036037,2.4042039720326818
0.49549549549549554,2.2283205464053752
0.6236236236236237,2.257678071872533
-0.7697697697697697,1.6140242420137656
0.6916916916916918,2.347125069451855
-0.08508508508508505,1.9078236562355713
-0.4814814814814815,1.807135359838545
-0.22522522522522526,1.8580416208459054
.....
.....
'''

# 直接使用数据集，进行训练
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('mydata.csv')
Y = np.array(df['Y_lbale'])
X = np.array(df['X_values'])
# plot data
plt.scatter(X, Y)
plt.show()


# 以上数据集和生成的完全一样，下一步我们做一下数据的预处理工作
%matplotlib inline
from keras.models import Sequential
from keras.layers import Dense, Activation
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pandas as pd
import matplotlib.pyplot as plt

X_train, Y_train = X[:800], Y[:800]     # 一共1000条数据,前800条数据用于训练
X_test, Y_test = X[800:], Y[800:]       # 后200条数据用于测试

# 经过预处理，我们发现已经是标准的  (X_train, y_train), (X_test, y_test) 数据集格式了
# 这个模型简单，我们搭建一个层就可以了
model = Sequential()
model.add(Dense(1,input_dim=1))
model.compile(loss='mse', optimizer='sgd')

print(model.summary())
SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg'))

model.fit(X_train, Y_train, epochs=300, batch_size=40,validation_data=(X_test, Y_test))
# test
print('\nTesting ------------')

W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 1)                 2         
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
None
Train on 800 samples, validate on 200 samples
Epoch 1/300
800/800 [==============================] - 0s - loss: 2.8850 - val_loss: 1.9322
Epoch 2/300
800/800 [==============================] - 0s - loss: 1.3227 - val_loss: 0.9019
Epoch 3/300
800/800 [==============================] - 0s - loss: 0.6181 - val_loss: 0.4313
Epoch 4/300
800/800 [==============================] - 0s - loss: 0.2979 - val_loss: 0.2139
Epoch 5/300
800/800 [==============================] - 0s - loss: 0.1503 - val_loss: 0.1112
Epoch 297/300
800/800 [==============================] - 0s - loss: 0.0025 - val_loss: 0.0020
Epoch 298/300
800/800 [==============================] - 0s - loss: 0.0025 - val_loss: 0.0020
Epoch 299/300
800/800 [==============================] - 0s - loss: 0.0025 - val_loss: 0.0020
Epoch 300/300
800/800 [==============================] - 0s - loss: 0.0025 - val_loss: 0.0020

Testing ------------
Weights= [[ 0.50460804]] 
biases= [ 1.99811506]
'''

## 我们把测试数据做一下预测，并画成图，直观地感受一下
# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()