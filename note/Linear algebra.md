# 标量
定义：只有一个元素的张量
``` py
x=tf.constant(1.0)
```

# 向量
用一维张量表示向量，长度任意.**列向量**是向量的默认方向
``` py
x=tf.range(4)
x[3]    #取向量的任意元素，是标量
```

# 长度、维度、形状
求张量长度
``` py
len(x)
x.shape
```

# 矩阵
- 求转置
``` py
tf.transpose(A)
```
- 对称矩阵
- Hadamard积：相同形状的张量对应元素相乘，数学符号为（一个圈里面一个点）
``` py
A*B
```

# 张量
描述具有任意数量轴的n维数组

# 降维
- 求和降维，沿某一条轴求和，结果中该轴消失
``` py
tf.reduce_sum(A,axis=0)    #按列求和，结果为1*n
tf.reduce_sum(A,axis=1)    #按行求和，结果为n*1
tf.reduce_sum(A,axis=[0,1])    #按行和列求和，结果为1*1，等同于tf.reduce_sum(A)
```
- 平均值
``` py
tf.reduce_mean(A)
tf.reduce_sum(A) / tf.size(A).numpy()
```
# 非降维求和
``` py
x=tf.reduce_sum(A,axis=1,keepdims=True) #结果保持原来的维度
```

# 点积
相同位置的按元素乘积的和
类似于先x*y，在reduce_sum
最总结果是一个元素
``` py
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)    #等同于tf.reduce_sum(x*y)
```
个人理解：假如x，y是4*1，x和y的点积是先x转置，变为1*4，然后x和y按照矩阵乘法相乘，结果为1*1。

# 矩阵-向量积
``` py
tf.linalg.matvec(A,x)    #shape:A:[5,4],x:[4] res[5,]
```

# 矩阵-矩阵乘法
``` py
tf.matmul(A,B)    #按照线性代数矩阵相乘的规则计算
```

# 范数
将向量映射到标量的函数
example：
- L2范数：元素平方和的平凡根
![图片](https://github.com/Sheibyer/Introduction-to-machine-learning/blob/16c604aa2fbe775f62544e721dda1f4e29196d33/picture/L2%E8%8C%83%E5%BC%8F.PNG "L2范式")
通常L2范式的下标不写，||x||==||x||2
``` py
tf.norm(u)
```
- L1范数：元素按绝对值求和
- ![图片](https://github.com/Sheibyer/Introduction-to-machine-learning/blob/fc7d98b6dca86f94f244c6e4cc1977f26a849d9c/picture/L1%E8%8C%83%E5%BC%8F.PNG)
``` py
tf.reduce_sum(tf.abs(u))
```
- Lp范数
- 矩阵的Frobenius范数
