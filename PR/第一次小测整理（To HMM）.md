### 基本的术语：

P(wi): the prior probability (先验概率）

P(X|H): the likelihood （似然）

P(X): the evidence probability 

P(wi|X): the posterior probability ( 后验概率)

### 基于贝叶斯的分类方法

本质的分类规则是**Bayes Decision Rule**，形式化描述如下
$$
\text { if } P\left(\omega_{j} | x\right)>P\left(\omega_{i} | x\right), \forall i \neq j \Longrightarrow \text { Decide } \omega_{j}
$$
但是我们是没有办法直接得到两个后验概率的，所以我们多从贝叶斯公式出发，考虑各个变量的关系进行判别

贝叶斯公式：
$$
P\left(\omega_{j} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \omega_{j}\right) \cdot P\left(\omega_{j}\right)}{p(\mathbf{x})}(1 \leq j \leq c)
$$
在大多数条件下，$p(x)$是相等的，所以我们只需要比较上面的$p\left(\mathbf{x} \mid \omega_{j}\right) \cdot P\left(\omega_{j}\right)$即可。

从原始的分类方法出发，我们有基于**minimum-error-rate**的分类方法，这种分类方法比较的是把具有某一特征的样本分到一个类之后产生的误差，并选择误差最小的分类。

我们看分类error的定义:
$$
P(\text { error } \mid x)=\left\{\begin{array}{ll}
P\left(\omega_{1} \mid x\right) & \text { if we decide } \omega_{2} \\
P\left(\omega_{2} \mid x\right) & \text { if we decide } \omega_{1}
\end{array}\right.
$$
我们要选择让错误的概率最小，那显然还是要选择两个后验概率中大的那个，最后的结果还是原来的比较$w_1w_2$的$p\left(\mathbf{x} \mid \omega_{j}\right) \cdot P\left(\omega_{j}\right)$的大小。

上面讨论的都是不考虑期望或者说风险或者说动作的条件下，关于这三者到有什么区别，可以去看老师的ppt，我觉得不是特别重要，反正就是不同的决策行为带来的后果是不一样的

就是两种办法。一个是：**minimum risk classification**还是拿两个类来说，多了反正也算不出来，假如我们要选1类，那我们要说明下面的式子成立
$$
R\left(\alpha_{1} \mid \mathbf{x}\right)<R\left(\alpha_{2} \mid \mathbf{x}\right)
$$
也就是计算：
$$
\begin{array}{r}
\lambda_{11} \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda_{12} \cdot P\left(\omega_{2} \mid \mathbf{x}\right) <

\lambda_{21} \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda_{22} \cdot P\left(\omega_{2} \mid \mathbf{x}\right)
\end{array}
$$
这里的后验概率就可以直接用先验概率以及似然代替。



还有一种办法就是用判别函数也就是：**Discriminant Function** 

如果我们要选择一个i类别的话也就是需要下面的式子满足
$$
\text { if } g_{i}(\mathbf{x})>g_{j}(\mathbf{x}) \text { for all } j \neq i
$$
题目中g多会直接给出，直接代入计算即可。



但是上面的方法有个问题是我们的似然函数不一定是一个定值，他很多时候是一个分布，只有当x确定下来的时候才是一个确定的值。比如说homework1的第3题,两个似然是给定的。
$$
P\left(x \mid \omega_{1}\right)=0.2, P\left(x \mid \omega_{2}\right)=0.4
$$
但是在第一题里面，似然就不一定了：
$$
p\left(x \mid \omega_{1}\right) \sim N(1,1), \quad p\left(x \mid \omega_{2}\right) \sim N(-1,1)
$$
并且在第四题里面，给了一个这个：
$$
p\left(\mathbf{x} \mid \omega_{3}\right)=\frac{1}{2} N\left(\left(\begin{array}{c}
0.5 \\
0.5
\end{array}\right), \mathbf{I}\right)+\frac{1}{2} N\left(\left(\begin{array}{c}
-0.5 \\
0.5
\end{array}\right), \mathbf{I}\right)   \ (1)
$$
乍一看是真的恶心，但是仔细想想，如果是传统的贝叶斯分类的话，当x给定的时候，先验，似然都有了，其实后验也直接出来了，那问题就变成了，当似然是（1）的时候，给定x要怎么算值呢？

首先后后面的$\sum=I$说明每个变量是不相关的，那么就直接加就行，就是简单的二次正态的公式:
$$
f_{X}(x) f_{Y}(y)=\frac{1}{2 \pi \sigma_{1} \sigma_{2}} e^{-\frac{\left(\mathfrak{q}-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}-\frac{\left(y-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}}
$$


### 最大似然和贝叶斯估计



**maximum-likelihood**

原理：通过对每个类进行一个最大似然的参数估计，然后对需要预测的样本，代入之前算好的两个估计当中，看哪边的概率较大。

大多数时候会假设似然服从高斯分布如第Homework2中第4题
$$
p\left(x \mid \omega_{1}\right) \sim N\left(\mu_{1}, \sigma_{1}^{2}\right), p\left(x \mid \omega_{2}\right) \sim N\left(\mu_{2}, \sigma_{2}^{2}\right)
$$
我们要做的就是估计上述的4个参数

参数估计计算：

1. 连乘

$$
p(\mathcal{D} \mid \boldsymbol{\theta})=\prod_{k=1}^{n} p\left(\boldsymbol{x}_{k} \mid \boldsymbol{\theta}\right)
$$

2. 取对数+对每一个参数求梯度,等于0 为最优参数估计

$$
0=\boldsymbol{\nabla}_{\boldsymbol{\theta}}\left(\sum_{k=1}^{n} \ln p\left(\mathbf{x}_{k} \mid \boldsymbol{\theta}\right)\right)
$$

判定类别：对于给定的$x=4$,计算$p(4|\theta_{1})$与$p(4|\theta_{2})$的大小



**Bayesian parameter estimation**

原理：通过已有的分布，推断出真实的分布，再代入值求解。

我们需要计算的结果是：
$$
\begin{array}{l}
p(x \mid \mathcal{D}) \sim 
N\left(\mu_{n}, \sigma^{2}+\sigma_{n}^{2}\right)
\end{array}
$$
我们有下述等式：
$$
\begin{aligned}
\sigma_{n}^{2} &=\frac{\sigma^{2} \sigma_{0}^{2}}{n \sigma_{0}^{2}+\sigma^{2}} \\
\mu_{n} &=\frac{\sigma_{n}^{2}}{\sigma^{2}} \sum_{k=1}^{n} x_{k}+\frac{\sigma_{n}^{2}}{\sigma_{0}^{2}} \mu_{0}
\end{aligned}
$$
所以对每一个$p(x \mid \mathcal{D})$

而我们又有：
$$
\begin{array}{l}
p(x \mid \mu) \sim N\left(\mu, \sigma^{2}\right) \\
p(\mu) \sim N\left(\mu_{0}, \sigma_{0}^{2}\right)
\end{array}
$$
我们需要的参数就是 $\sigma 、\sigma_{0}、\mu_{0}$这三个量都是已知的，比如说第4题里面：
$$
\sigma_{1}=1, \sigma_{2}=2, \mu_{1} \sim N(0,1) \text { and } \mu_{2} \sim N(3,1)
$$

对第一个类而言$\sigma_1=\sigma=1,\mu_0=0,\sigma_0^2=1$

那就直接可以代入计算了,我们可以直接算出$\mu_n\ \sigma_n^2$
$$
p(\mu \mid \mathcal{D}) \sim N\left(\mu_{n}, \sigma_{n}^{2}\right)
$$
最后根据：
$$
\begin{array}{l}
p(x \mid \mathcal{D}) \sim 
N\left(\mu_{n}, \sigma^{2}+\sigma_{n}^{2}\right)
\end{array}
$$
即可算出我们需要的分布

在预测的过程中，对$x=1.5$只需要比较$P(1.5|D_1)$与$P(1.5|D_2)$的大小即可,谁大选谁。



### 隐马尔科夫模型

主要考察三个计算方式：前向、后向、decoding用维特比。前两个用于估计给定序列估计概率，后两个用于给定表征求最大可能的状态。
下面给出一道例题进行讲解：

Given 
$$
\pi=
\left[
    \begin{array}{lll}
    0.1 & 0.3 & 0.6\\
    \end{array}
\right]
$$
$$
\mathbf{A}=
\left[
    \begin{array}{lll}
    0.2 & 0.2 & 0.4\\
    0.3 & 0.1 & 0.6\\
    0.5 & 0.2 & 0.3\\
    \end{array}
\right]
$$
$$
\mathbf{B}=
\left[
    \begin{array}{llll}
    0.2 & 0.5 & 0.1 & 0.2\\
    0.3 & 0.2 & 0.3 & 0.1\\
    0.4 & 0.4 & 0.1 & 0.1\\
    \end{array}
\right]
$$

1. Calculate $\alpha_2(2),\alpha_3(2)$ using forward algorithm given $\mathbf{V}^T=\{1,4,1\}$
2. Calculate $\beta_2(2)$ given $\mathbf{V}^T=\{1,4,1\}$
3. find $\omega^* = {argmax\atop{\omega}} p(\mathbf{\theta}|\mathbf{V}^T)$ given $\mathbf{V}^T=\{1,4,1\}$

预计考试的计算量不会超过这道题太多。

---
第一小题的计算过程按照形式化的语言描述如下：

$$
\alpha_j(t)=[\sum^c_{i=1}\alpha(t-1)a_{ij}]b_{jv(t)}
$$
$$
\alpha_j(1)=\pi_jb_{jv(1)}
$$

具体计算的时候可以记
$$
\alpha(t)=[\alpha_1(t),\alpha_2(t),\dots,\alpha_c(t)]
$$
作为一整个时间点上的全部结果。

首先求$\alpha(1)$：
$$
\alpha_1(1)=0.1*0.2=0.02\quad\alpha_2(1)=0.3*0.3=0.09\quad\alpha_3(1)=0.6*0.4=0.24\\
\alpha(1)=[0.02,0.09,0.24]
$$
用自然语言描述就是：给定特定隐状态$j$下在初始时间点1观察到给定序列中初始时间点1的特征的概率等于在该时间点1时处于隐状态$j$的概率乘以这个隐状态表现出状态$v(1)$的概率。

然后求$\alpha(2)$:
$$
\alpha_1(2)=(0.02*0.2+0.09*0.3+0.24*0.5)*0.2=0.0302\quad\alpha_2(2)=\dots\\
\alpha(2)=[0.0302,0.0061,0.0134]
$$
虽然题目中并没有提及$\alpha_1(2)$，但是为了说明这里也将其进行计算。

而后的计算在第三步中状态为2并且经历了$\mathbf{V}^T=\{1,4,1\}$表征状态的概率
$$
\alpha_{2}(3)=\left[\sum_{i=1}^{c} \alpha_{i}(2) a_{i 2}\right] b_{2 k}
$$
最后则是
$$
P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)=\sum_{j=1}^{c} \alpha_{j}(T)
$$

---



第二小题计算过程如下：

和第一题的过程是比较相似的，只是具体迭代的过程有细节上的差别。

形式化语言表示如下：
$$
\beta_j(T)=1\\
\beta_j(t)=\sum^c_{i=1}\beta_i(t+1)a_{ij}b_{iv(t+1)}\\
$$
此外，可以通过$\beta(1)$按照下面公式进一步求出$P(\mathbf{V}^T|\theta)$的概率：

$$
P(\mathbf{V}^T|\theta)=\sum^c_{i=1}\pi_jb_{jv(1)}\beta_j(1)
$$

$\beta(3)$属于已知：
$$
\beta(3)=[1,1,1]
$$

下面求$\beta(2)$:
$$
\beta_1(2)=1*0.2*0.2+1*0.2*0.3+1*0.4*0.4=0.26,\beta_2(2)=\dots\\
\beta(2)=[0.26,0.33,0.28]
$$

需要注意的是，这里使用转移概率其涵义从“从状态a转移到状态b的概率”变成了“状态b由状态a转移而来的概率”。

最终计算的概率为：
$$
P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)=\sum_{j=1}^{c} \pi_{j} b_{j v(1)} \beta_{j}(1)
$$




---

第三题的计算的思路比前两问复杂一些：原理是采用动态规划的办法去做。
$$
\delta_{j}(t)=\left[\max _{1 \leq i \leq c} \delta_{i}(t-1) a_{i j}\right] b_{j v(t)} ;
$$

$$
\psi_{j}(t)=\arg \max _{1 \leq i \leq c} \delta_{i}(t-1) a_{i j}
$$



有
$$
\psi_{i}(1)=0(1 \leq j \leq c)
$$

$$
\delta_{j}(1)=\pi_{j} b_{j v(1)}
$$



第一步可得
$$
\delta_1(1)=0.1*0.2=0.02\quad\delta_2(1)=0.3*0.3=0.09\quad\delta_3(2)=0.6*0.4=0.24\\
\delta(1)=[0.02,0.09,0.24]
$$

接下来计算$\delta(2)$与$\psi(1)$


计算可得
$$
\delta(2)=[0.024,0.0048,0.0072],\psi(1)=[3,3,3]\\
\delta(3)=[0.00096,0.00144,0.00384],\psi(2)=[1,1,1]\\
$$
所以
$$\omega^*=[3,1,3]$$

---
