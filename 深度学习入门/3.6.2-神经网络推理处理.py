import pickle

from mnist import load_mnist
import numpy as np


def getData():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True,one_hot_label=False)
    return x_test,t_test


def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax1(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3

    y =softmax1(a3)
    return y


x,t = getData()
network = init_network()
accuracy = 0

#普通写法
# for i in range(len(x)):
#     y = predict(network,x[i])
#     p = np.argmax(y)
#     if p==t[i]:
#         accuracy +=1


#批处理写法
batch_size = 100
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy += np.sum(p==t[i:i+batch_size])


print("Accuracy:"+str(float(accuracy)/len(x)))
