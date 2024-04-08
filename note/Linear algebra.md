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
tf.reduce_sum(A,axis=0)    #按行求和，结果为n*1
tf.reduce_sum(A,axis=[0,1])    #按行和列求和，结果为1*1，等同于tf.reduce_sum(A)
```
- 平均值
``` py
tf.reduce_mean(A)
tf.reduce_sum(A) / tf.size(A).numpy()
```
# 非降维求和
``` py
x=tf.reduce_sum(A,axis=1,keepdims=True) #结果保持两个维度
```


