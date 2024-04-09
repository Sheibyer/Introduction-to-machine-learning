# 自动微分
- 只适用于tf.Variable 或 tf.compat.v1.get_variable （相对于tf.constant）并设置为Trainable的变量可进行自动求导。
- 标量函数关于向量x的梯度是向量，且与x具有相同的形状
- 反向传播：填充关于每个参数的偏导数
eg：
``` py
# 把所有计算记录在磁带(t)上
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)

x_grad = t.gradient(y, x)  #反向传播函数自动计算y关于x每个分量的梯度
x_grad
```

# 非标量变量的反向传播
向量对向量的导数是矩阵

# 分离计算
- 对于**复合**函数，eg：z=f(y),y=f(x),计算z对于x的梯度时，有时我们想把中间量y看作常数，并且只考虑到x在被y计算后发挥的作用  ——>定义新变量u，使其与y由相同的值，但不会自动反向传播。
``` py
# 设置persistent=True来运行t.gradient多次
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u    #ans为<tf.Tensor: shape=(4,), dtype=bool, numpy=array([ True,  True,  True,  True])>，说明z = u * x中u是常数
``` 
