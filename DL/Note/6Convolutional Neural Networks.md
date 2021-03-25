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



# 6.3 Padding and Stride

在torch框架下，padding 的值=1意味着上下两边都+1

填充是为了，增加像素量，而不损失过多信息。保证图片输入输出一致，从而更好地预测模型。

卷积核常选为奇数，使得两边填充可以做到一致

步幅：每次移动多少格，处理原始的输入分辨率十分冗余的情况。



# 6.4  Multiple Input and Multiple Output Channels

通过减少空间分辨率以获得更大的通道深度。直观地说，我们可以将每个通道看作是对不同特征的响应



卷积核的形状$c_o\times c_i\times k_h\times k_w$



这种格式要会写：

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```



1x1的卷积核可以用全连接神经网络表示。



# 6.5 Pooling

降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性



矩阵合并：（类似刚刚上面的stack）

```
X = np.concatenate((X, X + 1), 1)
X
```

样本输入通道变多在合并的时候是增加第1维上：

```python
X = np.concatenate((X, X + 1), 1)
X
```

但是卷积核要输出多通道的时候是增加在第0维上：

```python
K = torch.stack((K, K + 1, K + 2), 0)
K.shape
```

# 6.6  Convolutional Neural Networks (LeNet)



