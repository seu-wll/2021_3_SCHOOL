# 6.1

1. In the** earliest layers**, our network should respond similarly to the same **patch**, regardless of where it appears in the image. This principle is called *translation invariance*.
2. The earliest layers of the network should focus on local regions, without regard for the contents of the image in distant regions. This is the *locality* principle. Eventually, these local representations can be aggregated to make predictions at the whole image level.

**上面两个关键词指什么？**

卷积的形式化表述公式，比较严谨，如果看不懂，暂时先跳过，不妨碍后面代码的理解。

$$
\begin{aligned}
[\mathbf{H}]_{i, j} &=[\mathbf{U}]_{i, j}+\sum_{k} \sum_{l}[\mathbf{W}]_{i, j, k, l}[\mathbf{X}]_{k, l} \\
&=[\mathbf{U}]_{i, j}+\sum_{a} \sum_{b}[\mathbf{V}]_{i, j, a, b}[\mathbf{X}]_{i+a, j+b} .
\end{aligned}
$$

### Translation Invariance

平移图像之后，所捕获的信息没有变化。

### Locality

捕获区域的范围是有限的。



# 6.2





**怎么理解这两个的区别，Cross-Correlation and Convolution**

flip the two-dimensional kernel tensor both horizontally and vertically

**关于核函数**
在案例中，有这样的代码：

```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
#K = torch.tensor([[1.0, -1.0],[-4,2]])
Y = corr2d(X, K)
```

此时通过:
```python
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
```
通过10轮学习可以成功收敛，最终训练出来的核函数参数与原来的一致。但是当我把神经网络的核函数维度设置为2x2的时候，就完全无法收敛，这是为什么，是正常现象吗？



**重要概念**


convolutional layer output=*feature map*

 *receptive field* = all the elements  (from all the previous layers) that may affect the calculation of x