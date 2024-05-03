空间不变性（spatial invariance）
#### 目录

> 1.[小结]()
>

### 6.1小结
1. 二维卷积层的核心计算是二维互相关运算。最简单的形式是，对二维输入数据和卷积核执行互相关操作，然后添加一个偏置。
2. 我们可以设计一个卷积核来检测图像的边缘。
3. 我们可以从数据中学习卷积核的参数。（通过反向传播更新学习参数）。
4. 学习卷积核时，无论用严格卷积运算或互相关运算，卷积层的输出不会受太大影响。
5. 当需要检测输入特征中更广区域时，我们可以构建一个更深的卷积网络。
6. 专业名称：**特征映射**：卷积层 ；**感受野**：在卷积神经网络中，对于某一层的任意元素x，其感受野（receptive field）是指在前向传播期间可能影响x计算的所有元素（来自所有先前层）。


### 6_1课后题

#### 构建一个具有对角线边缘的图像X。
``` py
X = torch.eye((8))
print("⼀个6 × 8像素的单位矩阵图像 : \n", X)
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print("互相关运算 : \n", Y)
print("corr2d(X.t(), K) : \n", corr2d(X.t(), K))
print("corr2d(X, K.t()) : \n", corr2d(X, K.t()))
```
1. 如果将本节中举例的卷积核K应用于X，会发生什么情况？
会有明显的过渡边界。

2. 如果转置X会发生什么？
结果与第一问一样

3. 如果转置K会发生什么？
形状变化，对角线变换存在且与第二问相反

#### 6.1.2在我们创建的Conv2D自动求导时，有什么错误消息？
无

##### 6.1.1如何通过改变输入张量和卷积核张量，将互相关运算表示为矩阵乘法？（ 题目的意思就是如何通过 矩阵乘法 得到 互相关（卷积）运算。）
``` py
def conv2d_by_mul(X, K):
    inh, inw = X.shape
    h, w = K.shape
    outh = inh - h + 1
    outw = inw - w + 1
    K = K.reshape(1, -1)
    XX = torch.zeros(K.shape[1], outh * outw)
    k = 0
    for i in range(outh):
        for j in range(outw):
            XX[:, k] = X[i:i + h, j:j + w].reshape(-1)
            k += 1
    # 用矩阵乘法表示互相关运算
    res = (torch.mm(K, XX)).reshape(outh, outw)
    return res


X = torch.randn((4, 4))
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))
print(conv2d_by_mul(X, K))
```

#### 6.1.4手工设计一些卷积核。
二阶导数的核的形式是什么？

积分的核的形式是什么？

得到d次导数的最小核的大小是多少？

参考链接[参考链接](https://dsp.stackexchange.com/questions/10605/kernels-to-compute-second-order-derivative-of-digital-image)
