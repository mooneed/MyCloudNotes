## 写在开头

产生写一个 pytorch 教程的念头的原因，其实是因为我在学习莫烦 pytorch 和动手学深度学习 2 pytorch 版时，所遇到的一个困扰：没有好用的可执行的 Python 文件。

莫烦大佬很厉害，讲的也很清晰，但是由于已经是 3-4 年前的创作了，而编程又是一个迭代很快的一个领域，故而其教程与现在版本存在一定的差异；动手学深度学习一书大而全，我觉得超级棒，代码也在 github 上有仓库，但是配套代码使用了太多自定义函数，结构也太过臃肿，很不利于作为一个小白的我来学习。故而我的创作，我希望是时下的，最有利于小白学习的 pytorch 教程。

## 预备知识

### 理论部分 && torch 库部分

预备知识请学习《动手学深度学习中文 pytorch 版》预备知识（章节），我已将pdf上传至仓库[点击这里](https://github.com/mooneed/MyCloudNotes/blob/main/%E5%88%9B%E4%BD%9C/pytorch/%E5%8A%A8%E6%89%8B%E5%AD%A6%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A02%E4%B8%AD%E6%96%87pytorch%E7%89%88.pdf)。值得注意的是书中代码块里的 d2l 库，即为我提到的，本书使用了太多自定义函数，造成了新手阅读代码的困难。所幸在预备知识章节，只有微分和概率两小节，含有极少量 matplotlib 绘图的自定义函数，可以选择性跳过，直接看书中展示的运行结果图即可。

阅读预备知识章节时，主要是对概念有一定的印象和初步的理解，千万不要死扣知识点。涉及张量 Tensor 的简单代码应该进行适当的编程实践。

### Python 部分

使用 pytorch 进行编程需要的 Python 知识有：

（1）Python 基础：关键字（部分仅了解）、运算符、字符串、列表（尤其是切片操作）、if 语句、for 语句、 函数`def`、迭代器与生成器（重点，pytorch的数据读取方式即以迭代器的方式进行）、输入输出、文件、异常处理、with 关键字（通常在异常小节内可以看到）。（元组、字典、集合仅了解）

（2）Python 高级：Python 面向对象（类与实例、继承，重写、类的内置函数（如，下文代码注释提到的 \__call__）等）、Python 语法糖（通俗理解为一些复杂代码的简便表达方式）

（3）Python 库：绘图库 matplotlib，数据分析库 pandas，数学计算库 numpy 等。（本部分可以随学随用，毕竟 pytorch 其实也只是一个库）

大部分教程可以从[菜鸟教程](https://www.runoob.com/python3/python3-tutorial.html)中学到，少部分需要百度自寻（部分关键字（如 with）、部分面向对象、语法糖等）。

## 简单实现线性回归

通过线性回归模型，了解使用 pytorch 训练一个模型所需要的各个模块。

### 模型及代码实现说明

数据集：本代码使用通过在真实数学表达式上增加噪声的方式生成了 1000 个散点数据作为数据集。

模型：y = W * X + b。 该数学表达式即为本代码的数学模型，其中W = [2, -3.4], b = 4.2。

损失函数：本代码使用均方误差作为损失函数，在复杂的神经网络模型中通常是使用交叉熵作为损失函数。

梯度下降算法：以模型的数学表达式外嵌套损失函数组成的复合函数为算法待优化的数学表达式，根据初始点位置梯度所指向方向前进，不断更新位置直到寻找到局部极小值，即为最终模型结果。

反向传播：反向传播算法是通过计算图以解析梯度计算法的方式，快速梯度值的算法（即反向传播是为加快梯度下降算法速度而设计的）。

数据可视化：通常包括数据集可视化和训练可视化，以及结果可视化。本代码仅包含数据集可视化。

### 具体代码

代码复制到 Pycharm 阅读效果更佳，代码提供了尽可能详细的注释。

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