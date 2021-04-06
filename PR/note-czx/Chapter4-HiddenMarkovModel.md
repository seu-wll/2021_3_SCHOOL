# Chapter 4 Hidden Markov Model
### First-Order Markov Model

#### Notations

- $\Omega=\left\{\omega_{1}, \omega_{2}, \ldots, \omega_{c}\right\}$: A set of $c$ possible states

- $\boldsymbol{\omega}^{T}=\{\omega(1), \omega(2), \ldots, \omega(T)\}$: A state sequence of Length $T$ where $w(t) \in \Omega$ , $1 \leq t \leq T$

  e.g. $\omega^{6}=\left\{\omega_{1}, \omega_{4}, \omega_{2}, \omega_{2}, \omega_{1}, \omega_{4}\right\}$

- $\mathbf{A}=\left[a_{i j}\right]_{c \times c}$: The transition probability matrix
  $$
  \left[\begin{array}{cccc}
  a_{11} & a_{12} & \cdots & a_{1 c} \\
  a_{21} & \cdots & \cdots & \cdots \\
  \cdots & \cdots & \cdots & \cdots \\
  a_{c 1} & \cdots & \cdots & a_{c c}
  \end{array}\right]
  $$
  
  where
  $$
  \begin{aligned}
  a_{i j} &=P\left(\omega(t+1)=\omega_{j} \mid \omega(t)=\omega_{i}\right) \\
  &=P\left(\omega_{j} \mid \omega_{i}\right)
  \end{aligned}
  $$
  
  probability of transferring from state $\omega_{i}$ to state $\omega_{j}$ is **time-independent**
  $$
  \sum_{j=1}^{c} a_{i j}=1
  $$

#### State transition diagram

<img src="images/image-20210331210437463.png" alt="image-20210331210437463" style="zoom: 33%;" />
$$
\begin{aligned}
P\left(\boldsymbol{\omega}^{T}\right) &=\prod_{t=1}^{T} P(\omega(t) \mid \omega(1), \ldots, \omega(t-1)) \\
&=\prod_{t=1}^{T} P(\omega(t) \mid \omega(t-1))
\end{aligned}
$$

### Hidden Markov Model (HMM)

#### Basic assumptions

- The state at each step is **invisible**
- The invisible state emits one **visible symbol** at each step

#### Notions

- $\mathcal{V}=\left\{v_{1}, v_{2}, \ldots, v_{K}\right\}:$ A set of $K$ possible symbols

- $\mathbf{V}^{T}=\{v(1), v(2), \ldots, v(T)\}:$ An observed symbol sequence of length $T$ where $v(t) \in \mathcal{V}(1 \leq t \leq T)$

- $\mathbf{B}=\left[b_{j k}\right]_{c \times K}:$ The observation symbol probability matrix
  $$
  \left[\begin{array}{cccc}
  b_{11} & b_{12} & \cdots & b_{1 K} \\
  \cdots & \cdots & \cdots & \cdots \\
  b_{c 1} & \cdots & \cdots & b_{c K}
  \end{array}\right]
  $$
  
  $$
  b_{j k}=P\left(v_{k} \mid \omega_{j}\right), \sum_{k=1}^{K} b_{j k}=1 \\
  $$
  which is the probability of emitting symbol $v_k$ at state $w_j$

- $\boldsymbol{\pi}=\left(\pi_{1}, \pi_{2}, \ldots, \pi_{c}\right)$: The initial state probability where $\pi_{j}=P\left(\omega(1)=\omega_{j}\right)$

#### State transition diagram

<img src="images/image-20210331210501439.png" alt="image-20210331210501439" style="zoom:33%;" />
$$
\begin{aligned}
P\left(\mathbf{V}^{T} \mid \boldsymbol{\omega}^{T}\right) &=\prod_{t=1}^{T} P(v(t) \mid \omega(t)) \\
&=\prod_{t=1}^{T} b_{\omega(t) v(t)}
\end{aligned}
$$

#### Three central problems in HMM

$$
\begin{aligned}
&\boldsymbol{\theta}=\{\mathbf{A}, \mathbf{B}, \boldsymbol{\pi}\}: \text { the complete set of } \mathrm{H}\\
&\mathbf{V}^{T}: \text { the observed symbol }
\end{aligned}
$$

##### Evaluation

Given $\boldsymbol{\theta},$ determine the probability of generating $\mathbf{V}^{T}$ **to evaluate $P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)$**

##### Decoding

Given $\boldsymbol{\theta}$ and $\mathrm{V}^{T}$, determine the most likely hidden state sequence **to identify $\boldsymbol{\omega}^{T}$ which maximizes** $P\left(\boldsymbol{\omega}^{T} \mid \mathbf{V}^{T}, \boldsymbol{\theta}\right)$

##### Learning

Given $\mathbf{V}^{T}$, determine model parameters $\boldsymbol{\theta}$ **to identify $\boldsymbol{\theta}$ which maximizes $P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)$**

#### Evaluation

##### HMM forward algorithm

Define $\alpha_{j}(t)=P\left(v(1), v(2), \ldots, v(t), \omega(t)=\omega_{j} \mid \boldsymbol{\theta}\right)$

which means the probability of being in hidden state $\omega_{j}$ at step $t$ and having generated the first $t$ symbols of $\mathbf{V}^{T}$

where the goal is $P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)=\sum_{j=1}^{c} \alpha_{j}(T)$

then calculate **recursively**:
$$
\begin{aligned}
\alpha_{j}(1)=P\left(v(1), \omega(1)=\omega_{j} \mid \boldsymbol{\theta}\right) &=P\left(\omega(1)=\omega_{j} \mid \boldsymbol{\theta}\right) \cdot P\left(v(1) \mid \omega_{j}, \boldsymbol{\theta}\right) \\
&=\pi_{j} b_{j v(1)}
\end{aligned}
$$

$$
\begin{aligned}
\alpha_{j}(t) &=\sum_{i=1}^{c} P\left(v(1), \ldots, v(t-1), \omega(t-1)=\omega_{i}, v(t), \omega(t)=\omega_{j} \mid \boldsymbol{\theta}\right) \\
&=\sum_{i=1}^{c} P\left(v(1), \ldots, v(t-1), \omega(t-1)=\omega_{i} \mid \boldsymbol{\theta}\right) \cdot P\left(\omega_{j} \mid \omega_{i}, \boldsymbol{\theta}\right) \cdot P\left(v(t) \mid \omega_{j}, \boldsymbol{\theta}\right) \\
&=\left[\sum_{i=1}^{c} \alpha_{i}(t-1) a_{i j}\right] b_{j v(t)}
\end{aligned}
$$

A trellis diagram (网格图):

<img src="images/image-20210331211115726.png" alt="image-20210331211115726" style="zoom: 50%;" />

e.g.

![image-20210331211330485](images/image-20210331211330485.png)

![image-20210331211342267](images/image-20210331211342267.png)

##### HMM backward algorithm

Define $\beta_{j}(t)=P\left(v(t+1), v(t+2), \ldots, v(T) \mid \omega(t)=\omega_{j}, \boldsymbol{\theta}\right)$

which means the probability of observing the rest $T-t$ symbols in $\mathbf{V}^{T}$ given that the hidden state at step t is $\omega_{j}$

where the goal is $P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)=\sum_{j=1}^{c} \pi_{j} b_{j v(1)} \beta_{j}(1)$

then calculate **recursively**:
$$
\beta_{j}(T)=1 \\
$$

$$
\begin{array}{l}
&\beta_{j}(t)&=\sum_{i=1}^{c} P\left(v(t+1), \omega(t+1)=\omega_{i}, v(t+2), \ldots, v(T) \mid \omega(t)=\omega_{j}, \boldsymbol{\theta}\right) \\
&&=\sum_{i=1}^{c} P\left(v(t+2), \ldots, v(T) \mid \omega(t+1)=\omega_{i}, \boldsymbol{\theta}\right) \cdot P\left(\omega_{i} \mid \omega_{j}, \boldsymbol{\theta}\right) \cdot P\left(v(t+1) \mid \omega_{i}, \boldsymbol{\theta}\right) \\
&&=\sum_{i=1}^{c} \beta_{i}(t+1) a_{j i} b_{i v(t+1)}
\end{array}
$$

#### Decoding

the goal is to find $\boldsymbol{\omega}^{*}=\arg \max _{\boldsymbol{\omega}^{T}} P\left(\boldsymbol{\omega}^{T} \mid \mathbf{V}^{T}, \boldsymbol{\theta}\right) =\arg \max _{\boldsymbol{\omega}^{T}} P\left(\boldsymbol{\omega}^{T}, \mathbf{V}^{T} \mid \boldsymbol{\theta}\right) \cdot P\left(\mathbf{V} \mid \boldsymbol{\theta}\right)$

because $P\left(\mathbf{V} \mid \boldsymbol{\theta}\right)$ is a constant to $\mathbf{\omega}$, the goal is the same to find
$$
\boldsymbol{\omega}^{*}=\arg \max _{\boldsymbol{\omega}^{T}} P\left(\boldsymbol{\omega}^{T}, \mathbf{V}^{T} \mid \boldsymbol{\theta}\right)
$$

##### The Viterbi algorithm

Define $\delta_{j}(t)=\max _{\omega(1), \ldots, \omega(t-1)} P\left(\omega(1), \ldots, \omega(t-1), \omega(t)=\omega_{j}, v(1), \ldots, v(t) \mid \boldsymbol{\theta}\right)$

which means **the highest probability (best score)** of the state sequence and observed symbols **till step** $\bold{t}$, where the state at step $t$ is $\omega_{j}$

(有关于$2t$个随机变量，将第$t$步的状态固定在$\omega_j$，遍历前$t-1$个状态共$c^{t-1}$次，找到使上式最大的$\omega(1), \ldots, \omega(t-1)$组合)

First calculate the first step:
$$
\begin{aligned}
\delta_{j}(1)=P\left(\omega(1)=\omega_{j}, v(1) \mid \boldsymbol{\theta}\right) &=P\left(\omega(1)=\omega_{j} \mid \boldsymbol{\theta}\right) \cdot P\left(v(1) \mid \omega_{j}, \boldsymbol{\theta}\right) \\
&=\pi_{j} b_{j v(1)}
\end{aligned}
$$
then
$$
\begin{aligned}
\delta_{j}(t) &=\max _{\omega(1), \ldots, \omega(t-1)} P\left(\omega(1), \ldots, \omega(t-1), \omega(t)=\omega_{j}, v(1), \ldots, v(t) \mid \boldsymbol{\theta}\right) \\
&=\max _{\omega(t-1)}\left[\max _{\omega(1), \ldots, \omega(t-2)} P\left(\omega(1), \ldots, \omega(t-1), \omega(t)=\omega_{j}, v(1), \ldots, v(t) \mid \boldsymbol{\theta}\right)\right] \\
&=\max _{1 \leq i \leq c}\left[\underset{\omega(1), \ldots, \omega(t-2)}{\max } P\left(\omega(1), \ldots, \omega(t-2), \omega(t-1)=\omega_{i}, v(1), \ldots, v(t-1) \mid \boldsymbol{\theta}\right)
\cdot P\left(\omega_{j} \mid \omega_{i}, \boldsymbol{\theta}\right) \cdot P\left(v(t) \mid \omega_{j}, \boldsymbol{\theta}\right)\right] \\
&=\left[\max _{1 \leq i \leq c} \delta_{i}(t-1) a_{i j}\right] b_{j v(t)}
\end{aligned}
$$
the pseudo-code

<img src="images/image-20210403102155812.png" alt="image-20210403102155812" style="zoom:50%;" />

where $\psi_{t+1}$ saves the max index $i^*$ of the step $t$

#### Learning

Generally, there is **no known algorithm** which can obtain the **optimal** solution to the above problem

Try to find a local optimum based on iterative updating: in each iteration, update $\boldsymbol{\theta}$ to $\hat{\boldsymbol{\theta}}$ such that $P\left(\mathbf{V}^{T} \mid \hat{\boldsymbol{\theta}}\right) \geq P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)$

##### The Baum-Welch algorithm

Define $\gamma_{i j}(t)=P\left(\omega(t)=\omega_{i}, \omega(t+1)=\omega_{j} \mid \mathbf{V}^{T}, \boldsymbol{\theta}\right)$

which means the probability of being in state $\omega_{i}$ at step $t$, and state $\omega_{j}$ at step $t+1$, given the observed symbol sequence

$$
\begin{aligned}
\gamma_{i j}(t) &=P\left(\omega(t)=\omega_{i}, \omega(t+1)=\omega_{j} \mid \mathbf{V}^{T}, \boldsymbol{\theta}\right) \\
&=\frac{P\left(\omega(t)=\omega_{i}, \omega(t+1)=\omega_{j}, \mathbf{V}^{T} \mid \boldsymbol{\theta}\right)}{P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)} \\
&=\frac{P\left(v(1), \ldots, v(t), \omega(t)=\omega_{i}, \omega(t+1)=\omega_{j}, v(t+1), \ldots, v(T) \mid \boldsymbol{\theta}\right)}{P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)} \\
&=\frac{\alpha_{i}(t) a_{i j} b_{j v(t+1)} \beta_{j}(t+1)}{P\left(\mathbf{V}^{T} \mid \boldsymbol{\theta}\right)} \\
&=\frac{\alpha_{i}(t) a_{i j} b_{j v(t+1)} \beta_{j}(t+1)}{\sum_{i=1}^{c} \sum_{j=1}^{c} \alpha_{i}(t) a_{i j} b_{j v(t+1)} \beta_{j}(t+1)}
\end{aligned}
$$

where

$$
\begin{array}
&\alpha_{i}(t)&=P\left(v(1), v(2), \ldots, v(t), \omega(t)=\omega_{i} \mid \boldsymbol{\theta}\right)\\
a_{ij}&=P\left(\omega_{j} \mid \omega_{i}\right)\\
b_{jv(t+1)}&=P\left(v(t+1) \mid \omega_{j}\right)\\
\beta_{j}(t+1)&=P\left(v(t+2), v(t+3), \ldots, v(T) \mid \omega(t+1)=\omega_{j}, \boldsymbol{\theta}\right)\\
\end{array}
$$
