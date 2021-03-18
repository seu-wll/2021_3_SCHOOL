# Soft Max

## 3.4.5. 小批量样本的矢量化

模型表达式：
$$
特征：\mathbf{X} \in \mathbb{R}^{n \times d} \\ 权重：\mathbf{W} \in \mathbb{R}^{d \times q} \\
偏置：\mathbf{b} \in \mathbb{R}^{1\times q}
$$

回归表达式里有广播的，不是直接矩阵加法。
$$
\begin{split}\begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned}\end{split}
$$

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

可以理解为一种后验概率，预测类别0,1被分类成0,2类的概率，那么就是分别为0.1,0.5