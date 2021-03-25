# 第一章

## 信号的分类

### 连续单位阶跃信号:
$$
\begin{array}{l}
u(t)=\left\{\begin{array}{ll}
1, & t>0 \\
0, & t<0
\end{array}\right. \\
x(t) u(t)=\left\{\begin{array}{cl}
x(t), & t>0 \\
0, & t<0
\end{array}\right.
\end{array}
$$

### 门信号:
$$
\begin{array}{l}
g(t)=\left\{\begin{array}{ll}
1, & |t|<\tau \\
0, & |t|>\tau
\end{array}\right. \\
g(t)=u(t+\tau)-u(t-\tau)
\end{array}
$$

### 单位脉冲信号
$$
\delta(n)=\left\{\begin{array}{ll}
1, & n=0 \\
0, & n \neq 0
\end{array}\right.
$$

### 单位脉冲信号性质
取样性：
$$
\begin{array}{l}
x(n) \delta(n)=x(0) \delta(n) \\
x(n) \delta(n-m)=x(m) \delta(n-m)
\end{array}
$$

### 单位阶跃序列和单位脉冲信号的关系
单位阶跃序列的一阶差分
$$
\delta(n)=u(n)-u(n-1)
$$

单位脉冲信号的求和
$$
u(n)=\sum_{k=0}^{\infty} \delta(n-k)
$$

### 单位冲激信号
定义：
$$
\left\{\begin{array}{c}
\int_{-\infty}^{\infty} \delta(t) d t=1 \\
\delta(t)=0 \quad t \neq 0
\end{array}\right.
$$

### 单位阶跃信号和单位冲激信号的关系
$$
\delta(t)=\frac{d u(t)}{d t} \quad \int_{-\infty}^{t} \delta(t) d t=u(t)
$$

### 单位冲激信号性质
抽样性质
$$
\begin{array}{l}
x(t) \delta(t)=x(0) \delta(t) \quad  \\
x(t) \delta\left(t-t_{0}\right)=x\left(t_{0}\right) \delta\left(t-t_{0}\right)
\end{array}
$$
偶函数
。。。
微分
$$
\int_{-\infty}^{\infty} x(t) \delta^{\prime}(t) d t=-x^{\prime}(0)
$$
积分
$$
\int_{-\infty}^{t} \delta(t) d t=u(t)
$$
尺度变换
$$
\delta(a t)=\frac{1}{|a|} \delta(t)
$$


## 系统的性质

### 即时系统与动态系统
即时系统:
$$
y(t)=k x(t) \quad y(n)=k x(n)
$$
动态系统:
$$
y(t)=x(t-1) \quad y(n)=\sum_{k=-\infty}^{n} x(n)
$$
恒等系统 
$$
\quad y(t)=x(t)
$$
### 系统的可逆性与逆系统
输入、输出是否一一对应

### 系统的因果性
用没有用t+1

### 时变与时不变系统
时不变系统
$$
\begin{array}{ll}
x(t) \rightarrow y(t) & x\left(t-t_{0}\right) \rightarrow y\left(t-t_{0}\right) \\
x(n) \rightarrow y(n) & x\left(n-n_{0}\right) \rightarrow y\left(n-n_{0}\right)
\end{array}
$$

### 线性与非线性系统
* 齐次性与叠加性
* 零输入零输出的特性为必要条件
$$
k_{1} x_{1}(t)+k_{2} x_{2}(t) \rightarrow k_{1} y_{1}(t)+k_{2} y_{2}(t)
$$

# 第二章

## 离散时域分析

### 卷积和
$$
y(n)=\sum_{k=-\infty}^{+\infty} x(k) h(n-k)=x(n)^{*} h(n)
$$

### 卷积和的性质
交换律
$$
x(n) * h(n)=h(n)^{*} x(n)
$$
结合律
$$
\left[x(n) * h_{1}(n)\right] * h_{2}(n)=x(n) *\left[h_{1}(n)^{*} h_{2}(n)\right]
$$
分配律
$$
x(n) *\left[h_{1}(n)+h_{2}(n)\right]=x(n) * h_{1}(n)+x(n)^{*} h_{2}(n)
$$
时移性质
$$
x\left(n-n_{0}\right) * h(n)=x(n) * h\left(n-n_{0}\right)=y\left(n-n_{0}\right)
$$
差分性质
$$
[x(n)-x(n-1)] * h(n)=y(n)-y(n-1)
$$
求和性质
$$
\left[\sum_{k=-\infty}^{n} x(k)\right] * h(n)=\sum_{k=-\infty}^{n} y(k)
$$

其他
$$
\begin{aligned}
x(n) * \delta(n) &=x(n) & x\left(n-n_{1}\right) * \delta\left(n-n_{2}\right)=x\left(n-n_{1}-n_{2}\right) \\
x(n) * u(n) &=\sum_{k=-\infty}^{n} x(k) &
\end{aligned}
$$

## 连续时域分析
### 卷积和
定义
$$
f_{1}(t) * f_{2}(t)=\int_{-\infty}^{\infty} f_{1}(\tau) f_{2}(t-\tau) d \tau
$$
### 卷积和的性质

$$
\begin{array}{ll}
\text { 微分: } & \frac{d}{d t}\left[f(t)^{*} v(t)\right]=\left[\frac{d}{d t} f(t)\right] * v(t)=f(t) *\left[\frac{d}{d t} v(t)\right] \\
\text { 积分: } & \int_{-\infty}^{t}[f(\tau) * v(\tau)] d \tau=f(t) *\left[\int_{-\infty}^{t} v(\tau) d \tau\right]=\left[\int_{-\infty}^{t} f(\tau) d \tau\right] * v(t) \\
& \int_{-\infty}^{t} f(\tau) d \tau * \frac{d v(t)}{d t}=\frac{d}{d t}\left(\int_{-\infty}^{t} f(\tau) d \tau\right) * v(\tau)=f(t) * v(t) \\
& \frac{d f(t)}{d t} * \int_{-\infty}^{t} v(\tau) d \tau=f(t) * \frac{d}{d t}\left(\int_{-\infty}^{t} v(\tau) d \tau\right)=f(t) * v(t) \\
\text { 推论: } & f(t) * v(t)=\int_{-\infty}^{t} f(\tau) d \tau * \frac{d v(t)}{d t}=\frac{d f(t)}{d t} * \int_{-\infty}^{t} v(\tau) d \tau
\end{array}
$$

### 常用函数的卷积

(1) $f(t)^{*} \delta(t)=f(t)$
(2) $f(t)^{*} \delta\left(t-t_{0}\right)=f\left(t-t_{0}\right)$
(3) $f(t) * \delta^{\prime}(t)=f^{\prime}(t) * \delta(t)=f^{\prime}(t)$
(4) $f(t) * u(t)=\int_{-\infty}^{t} f(\tau) d \tau$
$(5) u(t) * u(t)=t u(t)$
(6) $e^{-t} u(t) * u(t)=\left(1-e^{-t}\right) u(t)$
