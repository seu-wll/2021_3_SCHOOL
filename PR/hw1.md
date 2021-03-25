# 第一章解题相关

### 数学知识：

1. 对同样的分布而言，若在该点的概率一样，则两点的大小一致，但是概率密度不具有这样的性质。
2. 高斯分布，若两个变量无关，则可以较为方便地求出概率密度，如t4. $\Sigma$矩阵一致，且变量无关。则可以直接转化为指数项的加和进行计算。
$$
f(X)=\frac{1}{\left.(2 \pi)^{d / 2} | \Sigma\right|^{1 / 2}} \exp \left[-\frac{1}{2}(X-u)^{T} \Sigma^{-1}(X-u)\right]
$$



### 最大最小原则：

书本推导不清晰，如果有考也只是直接用定义去做。

$$
0=\left(\lambda_{11}-\lambda_{22}\right)+\left(\lambda_{21}-\lambda_{11}\right) \int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}-\left(\lambda_{12}-\lambda_{22}\right) \int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) d \mathbf{x}
$$
根据左边这个式子，可以推导出$\int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}$ 和 $\int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) d \mathbf{x}$ 的关系，进而通过将第二类分类映射到第一类分类上可以解决问题。t1



### 贝叶斯决策

就是看谁的后验概率比较小了，就直接拿这个去算

$$
P(\omega \mid x)=\frac{p(x \mid \omega) \cdot P(\omega)}{p(x)} \text { (Bayes Formula) }
$$



### 最小分类误差：

$$
P(\text { error } \mid x)=\min \left[P\left(\omega_{1} \mid x\right), P\left(\omega_{2} \mid x\right)\right]
$$

这个是二分类的情况，三分类的话就是：
$$
P(\text { error } \mid x)=\min \left[1-P\left(\omega_{1} \mid x\right)- P\left(\omega_{2} \mid x\right),1-P\left(\omega_{2} \mid x\right)\\- P\left(\omega_{3} \mid x\right),1-P\left(\omega_{1} \mid x\right)- P\left(\omega_{3} \mid x\right)\right]
$$
