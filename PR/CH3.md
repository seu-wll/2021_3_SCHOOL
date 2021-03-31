# 浅析隐马尔科夫模型的公式与实例

结合《模式识别第二版》的公式+知乎用户@henry的回答[如何用简单易懂的例子解释隐马尔可夫模型？ - henry的回答 - 知乎 ](https://www.zhihu.com/question/20962240/answer/64187492)

### 以女朋友&天气实例为例。

首先定义状态，状态的形式化表述：
$$
\Omega=\left\{\omega_{1}, \omega_{2}, \ldots, \omega_{c}\right\}
$$
我们假设c=3，那么天气就是三个状态，晴天，雨天，多云。然后定义状态序列：
$$
\boldsymbol{\omega}^{T}=\{\omega(1), \omega(2), \ldots, \omega(T)\}
$$

其中T代表了时间，可以理解为在T天的时间里面，东南大学的天气状态{多云，雨天，多云...}

马尔科夫是干啥用的，是来预测时间的嘛，那就必然是有一个预测矩阵在那里，表示一个状态转移到下一个状态。我们在这里用状态转移矩阵来表述。
$$
\mathbf{A}=\left[a_{i j}\right]_{c \times c}
$$
其中
$$
\begin{aligned}
a_{i j} &=P\left(\omega(t+1)=\omega_{j} \mid \omega(t)=\omega_{i}\right) \\
&=P\left(\omega_{j} \mid \omega_{i}\right)
\end{aligned}
$$
表示了从一个状态转移到另外一个状态的概率，这里暂时忽略了时间，所以可以简写，大家知道是这么表示就行。

因此我们有状态转移矩阵的描述:
$$
\left[\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 c} \\
a_{21} & \cdots & \cdots & \cdots \\
\cdots & \cdots & \cdots & \cdots \\
a_{c 1} & \cdots & \cdots & a_{c c}
\end{array}\right]
$$
显然:
$$
\sum_{j=1}^{c} a_{i j}=1
$$
状态转移矩阵的行和为零，直观的理解便是不管概率怎么转移，状态是一定要转移的，那么转移过去的状态之和必然为零。



懒得写了。



从后面开始讲起吧。

