## pytorch 常用组件介绍

### Dataset

### Dataloader

### Module

## 简单实现线性回归

代码复制到 Pycharm 阅读效果更佳。

```
"""
@Time: 2021/9/16 10:20
@Auth: 边缘
@Email: 785314906@qq.com
@Project: Demo_study
@IDE: PyCharm
@Motto: I love my country as much as myself.
"""
import torch
from matplotlib import pyplot as plt
from IPython import display
import random

# 设置数据可视化图片尺寸和格式
def set_figsize(figsize=(19.60, 14.70)):
    #display.set_matplotlib_formats('svg')# 似乎不生效
    plt.rcParams['figure.figsize'] = figsize# rcParams基本可以修改图的任何属性

# 生成数据
def synthetic_data(w, b, num_examples):
    # ⽣成 y = Xw + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w)))# 正态分布N(0,1)的随机采样
    #print("X size:", X.size())# torch.Size([1000, 2])
    #print("w size:", w.size())# torch.Size([2])
    y = torch.matmul(X, w) + b# X形状(1000,2)，w形状(2)自动扩展到(2,1)
                              # ，矩阵乘法（会自动调整矩阵的来适应吗？毕竟没有转置）
    y += torch.normal(0, 0.01, y.shape)# 增加噪声
    return X, y.reshape((-1, 1))# y通过reshape提升了一个维度
                                # ，每个元素都是一个小tensor内含一个y值

# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]# generator：执行到yield时返回一个迭代值
                                                            # ，下一次调用时继续上次的循环位置执行

# 模型
def linreg(X, w, b):# 线性回归模型
    return torch.matmul(X, w) + b

# 损失函数
def squared_loss(y_hat, y):# 均⽅误差
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 梯度下降优化器
def sgd(params, lr, batch_size):
    # ⼩批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size# 为啥要除batch_size？
            param.grad.zero_()# 清空梯度值

if __name__ == '__main__':
    # 生成数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])
    # 数据可视化
    set_figsize()
    plt.scatter(features[:, 1].numpy(),#Tensor对象可以通过Python索引和切片进行操作
                labels.numpy(),
                s = 100# 散点图中数据点的大小
                )
    plt.savefig('Linear_scatter_synthetic_dataset.svg')# 保存必须在展示前面，否则保存的是空白文件
    plt.show()
    # 读取小批量数据
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        break
        print(X, '\n', y)
    # 模型参数初始化
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # 训练参数初始化
    lr = 0.03
    num_epochs = 3
    net = linreg# Python支持直接把函数的定义赋值给变量
    loss = squared_loss
    # 训练(只有前向传播的tensor计算需要构建计算图)
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            y_predict = net(X, w, b)# 前向传播，计算预测值
                                    # 通过Python语言特性__call__直接使用实例名作为函数名
                                    # ，可执行代码可在该类的__call__函数出查看
            l = loss(y_predict, y)# 计算小批量损失值
    l.sum().backward()# 因为l形状是(batch_size, 1)，⽽不是⼀个标量。
                      # 所以需要把l中的所有元素被加到⼀起，并以此计算关于[w, b]的梯度。为什么？？？？？？？？
    sgd([w, b], lr, batch_size)# 使⽤参数的梯度更新参数
    with torch.no_grad():# 该语句下的代码块的tensor计算不构建计算图
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差: {(true_b - b)}')
```