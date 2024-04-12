> 前言：由于本人较为熟悉TensorFlow框架，故一下均为tf语言。
# 创建张量
1. 创建一维张量
```  py
 x=tf.range(12)
 ```
2. 获取张量的形状
``` py
x.shape
```
3. 获得张量的元素个数，若张量为一维，则shape与size相同
``` py
tf.size(x)
```
4.改变张量的形状（元素个数不变）
``` py
X=tf.reshape(x,(3,4))
```
通过-1自动计算reshape后张量的维度
``` py
X=tf.reshape(x,(3,-1))
```
5.创建元素全为0的张量
``` py
tf.zeros((1,2,3))
```
6. 创建元素全为1的张量
``` py
tf.ones((1,2,3))
```
7.创建张量，元素满足特定概率分布，示例为均值为0，标准差为1的正态分布。  **通常用来构造数组作为神经网络的参数**
``` py
tf.random.normal(shape=[3,4])
```
8. 自定义张量元素的值
``` py
tf.constant([[1,2],[3,4]])
```
# 运算符
1. **相同形状**的张量可进行+ - * / ** 等基本运算，**对应元素**作运算
2. 一元运算符 求幂
``` py
tf.exp(x)
```
3. 多个张量**连结**，按照**某个维度**合并
``` py
tf.concat([tensor1,tensor2],axis=0))
```
4. **逻辑运算符**构建二维张量，前提：**张量的形状相同**，新张量与原来相同，元素为0（两张量对应元素相同）/1（两张量对应元素不同）
``` py
X==Y
```
5. 对张量所有元素求和
``` py
tf.reduce_sum(x)
```

# 广播机制
*适用情况：进行操作的两个张量形状不同，需要改成相同形状* 一般要拓展张量维度，且拓展长度为1的维度,该维度上的值相同。
``` py
a=tf.reshape(x,(3,1))
b=tf.reshape(y,(1,2))
a+b
```
结果张量的形状为（3，2）

# 索引和切片
等同于python数组的索引和切片，目的是获取张量的部分张量元素
**注意：Tensors是不可变的，也不可被赋值**     Variables可以被赋值    **TensorFlow中的梯度不会通过Variables反向传播**
- 单个元素赋值
``` py
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```
结果张量（1，2）位置的元素被替换为9

- 部分元素赋值

# 内存使用
TensorFlow没有提供一种明确的方式来**原地**运行单个操作，所以，y=y+x，等号左右两边y的地址不同。->会导致额外的内存开销
解决方法：
- assign
``` py
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))

```
z的地址没有发生改变
- @tf.function

# ndarray和tensor相互转化
- ndarray->tensor
``` py
b=tf.constant(x) #x是数组
```
- tensor->ndarray
``` py
a=tf.constant([1,2]).numpy()
```

