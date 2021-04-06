# Chapter 2 Bayesian Decision Theory

### Bayesian Decision Theory

Pattern Recognition is a decision process in essence

Bayesian decision theory is a **statistical approach** to pattern recognition

##### Basic Assumptions

- The decision problem is posed (formalized) in probabilistic terms
- All the relevant probability values are known

##### Key Principle: Bayes Theorem

$$
P(H | X)=\frac{P(H) P(X | H)}{P(X)}
$$

$$
P\left(\omega_{j} | x\right)=\frac{p\left(x | \omega_{j}\right) \cdot P\left(\omega_{j}\right)}{p(x)}
$$

$$
\text { posterior }=\frac{\text { likelihood } \times \text { prior }}{\text { evidence }}
$$

- $X$: the observed sample / **evidence **
  e.g. the length of a fish
-  $H$: the **hypothesis**
  e.g. the fish belongs to the “salmon” category
- $P(H)$: the **prior probability** (先验概率) that $H$ holds 
  e.g. the probability of catching a salmon
- $P(X|H)$: the **likelihood** (似然度) of observing $X$ given that H holds
  e.g. the probability of observing a 3-inch length fish which is salmon
- $P(X)$: the **evidence probability** that $X$ is observed
  e.g. the probability of observing a fish with 3-inch length
- $P(H|X)$: the **posterior probability** (后验概率) that $H$ holds given $X$
  e.g. the probability of $X$ being salmon given its length is 3-inch

##### State of Nature

- Future events that might occur

- State of nature is unpredictable
- Regarded as a random variable

e.g. let $w$ denote the (discrete) random variable representing the **state of nature (class)** of fish types

​		$w = w_1$: sea bass / $w = w_2$: salmon

##### Prior Probability

Prior probability is the probability distribution which **reflects one’s prior knowledge on the random variable**

- the catch produced as much sea bass as salmon $\rightarrow P(w_1) = P(w_2) = \frac{1}{2}$
- the catch produced more sea bass than salmon $\rightarrow P(w_1) = \frac{2}{3}; P(w_2) = \frac{1}{3}$

##### Class-conditional probability density function

It is a probability density function (pdf) for $x$ given that the state of nature (class) is $\omega$
$$
p(x | \omega) \geq 0 \quad \int_{-\infty}^{\infty} p(x | \omega) d x=1
$$
The class-conditional pdf describes the difference in the distribution of observations under different classes
$$
p\left(x | \omega_{1}\right) \text { should be different to } p\left(x | \omega_{2}\right)
$$

##### Decision After Observation

<img src="images/image-20210308100648341.png" alt="image-20210308100648341" style="zoom: 33%;" />

##### Bayes Decision Rule

$$
\text { if } P\left(\omega_{j} | x\right)>P\left(\omega_{i} | x\right), \forall i \neq j \Longrightarrow \text { Decide } \omega_{j}
$$

- $P\left(\omega_{j}\right)$ and $p\left(x | \omega_{j}\right)$ are assumed to be known

- $p(x)$ is **irrelevant** for Bayesian decision but **serving as a normalization factor** and not related to any state of nature)

##### Two Special Cases

- Equal prior probability: Depends on the likelihood $P\left(x | \omega_{j}\right)$

$$
P\left(\omega_{1}\right)=P\left(\omega_{2}\right)=\cdots=P\left(\omega_{c}\right)=\frac{1}{c}
$$

- Equal likelihood: Degenerate to naïve decision rule
  $$
  p\left(x | \omega_{1}\right)=p\left(x | \omega_{2}\right)=\cdots=p\left(x | \omega_{c}\right)
  $$

##### Example: Cancer or not?

Problem statement

- A new medical test is used to detect whether a patient has a certain cancer or not, whose test result is either + positive ) or negative
- For patient with this cancer, the probability of returning positive test result is 0.98
- For patient without this cancer, the probability of returning negative
test result is 0.97
- The probability for any person to have this cancer is 0.008

Q: If positive test result is returned for some person, does he/she have this kind of cancer or not?

Solution:

Define the notation:
$$
\omega_{1}: \text { cancer } \quad \omega_{2}: \text { no cancer } \quad x \in\{+,-\}
$$

Calculate likelihood:
$$
\begin{array}{ll}
P\left(\omega_{1}\right)=0.008 & P\left(\omega_{2}\right)=1-P\left(\omega_{1}\right)=0.992 \\
P\left(+\mid \omega_{1}\right)=0.98 & P\left(-\mid \omega_{1}\right)=1-P\left(+\mid \omega_{1}\right)=0.02 \\
P\left(-\mid \omega_{2}\right)=0.97 & P\left(+\mid \omega_{2}\right)=1-P\left(-\mid \omega_{2}\right)=0.03
\end{array}
$$

Calculate posterior probability:
$$
P\left(\omega_{1} \mid+\right)=\frac{P\left(\omega_{1}\right) P\left(+\mid \omega_{1}\right)}{P(+)}=\frac{P\left(\omega_{1}\right) P\left(+\mid \omega_{1}\right)}{P\left(\omega_{1}\right) P\left(+\mid \omega_{1}\right)+P\left(\omega_{2}\right) P\left(+\mid \omega_{2}\right)} = 0.2085
$$

$$
P\left(\omega_{2} \mid+\right)=1-P\left(\omega_{1} \mid+\right)=0.7915
$$

We find out that
$$
P\left(\omega_{2} \mid+\right)>P\left(\omega_{1} \mid+\right)
$$
So he/she have no cancer!

##### Feasibility of Bayes Formula

Get to know the Prior probability $P(w)$ and likelihood $p(x \mid w)$

- Counting relative frequencies (相对频率)
  e.g. Suppose we have randomly picked 1209 cars in the campus, got prices from their owners, and measured their heights
  $$
  cars \ in \ w_1 = 221 \longrightarrow P\left(\omega_{1}\right)=\frac{221}{1209}=0.183 \\
  cars \ in \ w_2 = 988 \longrightarrow P\left(\omega_{1}\right)=\frac{988}{1209}=0.817 \\
  $$

- Conduct density estimation (概率密度估计)
  **Discretize** the height spectrum (say [0.5m, 2.5m]) into 20 intervals each with length 0.1m, and then count the number of cars falling into each interval for either class

  Suppose $x = 1.05$，and $x$ falls into interval $I_x = [1.0m,1.1m]
For $\omega_{1}$, cars in $I_{x}$ is 46; For $\omega_{2}$, cars in $I_{x}$ is 59
  So
  $$
  p(x=1.05 \mid w_1) = \frac{46}{221} = 0.2081 \\
  p(x=1.05 \mid w_2) = \frac{59}{988} = 0.0597
  $$
  

##### Is Bayes Decision Rule Optimal?

Bayes Decision Rule (In case of two classes)
$$
\text { if } P\left(\omega_{1} \mid x\right)>P\left(\omega_{2} \mid x\right) \text { , Decide } \omega_{1} \text { ; Otherwise } \omega_{2}
$$
Whenever we observe a particular x, the probability of error is:
$$
P(\text { error } \mid x)=\left\{\begin{array}{ll}
P\left(\omega_{1} \mid x\right) & \text { if we decide } \omega_{2} \\
P\left(\omega_{2} \mid x\right) & \text { if we decide } \omega_{1}
\end{array}\right.
$$
Under Bayes decision rule, we have
$$
P(\text { error } \mid x)=\min \left[P\left(\omega_{1} \mid x\right), P\left(\omega_{2} \mid x\right)\right]
$$
The **average probability of error** over all possible x must be **as small as possible**

##### The General Case of Bays Decision Rule

- By allowing to **use more than one feature** (d-dimensional Euclidean space)
  $$
  x \in \mathbf{R} \Longrightarrow \mathbf{x} \in \mathbf{R}^{d}
  $$

- By allowing more than two states of nature (finite set of c states of nature)
  $$
  \Omega=\left\{\omega_{1}, \omega_{2}, \ldots, \omega_{c}\right\}
  $$

- By **allowing actions other than merely deciding the state of nature** (finite set of a possible actions)
  $$
  \mathcal{A}=\left\{\alpha_{1}, \alpha_{2}, \ldots, \alpha_{a}\right\}
  $$
  Note that $c \neq a$

##### Loss function

$\lambda(w_j,\alpha_j)$ [also written as $\lambda (\alpha_i \mid w_j)$] is the loss incurred for taking action $\alpha_i$ when the state of nature is $w_j$
$$
\lambda : \Omega \times \mathcal{A} \rightarrow R
$$
<img src="images/image-20210308111436298.png" alt="image-20210308111436298" style="zoom: 50%;" />

##### Action

Given a particular $x$, we have to decide which **action** to take.

With the loss of taking each action $\alpha_i$, to minimize the loss $\lambda(\alpha_i \mid w_j)$

However, the true state of nature is uncertain: **Expected (average) loss**

##### Expected loss (Conditional risk)

Average by **enumerating** over all possible states of nature
$$
R\left(\alpha_{i} \mid \mathbf{x}\right)=\sum_{j=1}^{c} \lambda\left(\alpha_{i} \mid \omega_{j}\right) \cdot P\left(\omega_{j} \mid \mathbf{x}\right)
$$
where $\lambda\left(\alpha_{i} \mid \omega_{j}\right)$ is the incurred loss of taking action $\alpha_i$ in case of true state of nature being $w_j$, and $P\left(\omega_{j} \mid \mathbf{x}\right)$ is the probability of $w_j$ being the true state of nature.

e.g.
$$
\begin{aligned}
R\left(\alpha_{1} \mid \mathbf{x}\right) &=\sum_{j=1}^{2} \lambda\left(\alpha_{1} \mid \omega_{j}\right) \cdot P\left(\omega_{j} \mid \mathbf{x}\right) \\
&=\lambda\left(\alpha_{1} \mid \omega_{1}\right) \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda\left(\alpha_{1} \mid \omega_{2}\right) \cdot P\left(\omega_{2} \mid \mathbf{x}\right) \\
&=5 \times 0.01+60 \times 0.99=59.45
\end{aligned}
$$

##### Task

find a mapping form patterns to actions
$$
\alpha: \mathbf{R}^{d} \rightarrow \mathcal{A} \text { (decision function) }
$$
In other words, for every $\mathbf{x}$, the decision function $\alpha(\mathbf{x})$ assumes one of the $a$ actions $\{ \alpha_{1}, \ldots, \alpha_{a} \}$

##### Overall risk R

expected loss with decision function $\alpha(\cdot)$
$$
R=\int R(\alpha(\mathbf{x}) \mid \mathbf{x}) \cdot p(\mathbf{x}) d \mathbf{x}
$$
where $p(\mathbf{x})$ is the pdf for pattern

For every $\mathbf{x}$, we ensure that the conditional risk $R(\alpha(\mathbf{x}) \mid \mathbf{x})$ is as small as possible, so the risk over all possible $\mathbf{x}$ must be as small as possible.
$$
\begin{aligned}
\alpha(\mathbf{x}) &=\arg \min _{\alpha_{i} \in \mathcal{A}} R\left(\alpha_{i} \mid \mathbf{x}\right) \\
&=\arg \min _{\alpha_{i} \in \mathcal{A}} \sum_{j=1}^{c} \lambda\left(\alpha_{i} \mid \omega_{j}\right) \cdot P\left(\omega_{j} \mid \mathbf{x}\right)
\end{aligned}
$$
The resulting overall risk is called the **Bayes risk** $\left(\right.$ denoted as $\left.R^{*}\right)$, which is the **best performance** achievable given $p(\mathbf{x})$ and loss function.

##### Example 1: Two-Category Classification

Classification setting

- $\Omega = \{ w_1, w_2 \}$: two states of nature
- $\mathcal{A}=\left\{\alpha_{1}, \alpha_{2}\right\}$: $\alpha_1$ means decide $w_1$, $\alpha_2$ means decide $w_2$
- $\lambda_{ij} = \lambda(\alpha_i \mid w_j)$: the loss incurred for deciding $w_i$ when the true state of nature is $w_j$

The conditional risk
$$
\begin{array}{l}
R\left(\alpha_{1} \mid \mathbf{x}\right)=\lambda_{11} \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda_{12} \cdot P\left(\omega_{2} \mid \mathbf{x}\right) \\
R\left(\alpha_{2} \mid \mathbf{x}\right)=\lambda_{21} \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda_{22} \cdot P\left(\omega_{2} \mid \mathbf{x}\right)
\end{array}
$$
If we assume that
$$
R\left(\alpha_{1} \mid \mathbf{x}\right) < R\left(\alpha_{2} \mid \mathbf{x}\right)
$$
by definition we get
$$
\lambda_{11} \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda_{12} \cdot P\left(\omega_{2} \mid \mathbf{x}\right) < \lambda_{21} \cdot P\left(\omega_{1} \mid \mathbf{x}\right)+\lambda_{22} \cdot P\left(\omega_{2} \mid \mathbf{x}\right)
$$
by rearrangement we get
$$
\left(\lambda_{21}-\lambda_{11}\right) P\left(\omega_{1} \mid \mathbf{x}\right) >
\left(\lambda_{12}-\lambda_{22}\right) P\left(\omega_{2} \mid \mathbf{x}\right)
$$
by Bayes theorem we get
$$
\left(\lambda_{21}-\lambda_{11}\right) p\left(\mathbf{x} \mid \omega_{1}\right) \cdot P\left(\omega_{1}\right) >
\left(\lambda_{12}-\lambda_{22}\right) p\left(\mathbf{x} \mid \omega_{2}\right) \cdot P\left(\omega_{2}\right)
$$
because $\lambda_{21} - \lambda_{11} > 0$, which means the loss for being error is ordinarily greater than the loss for being correct, we get
$$
\frac{p\left(\mathbf{x} \mid \omega_{1}\right)}{p\left(\mathbf{x} \mid \omega_{2}\right)}>\frac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}} \cdot \frac{P\left(\omega_{2}\right)}{P\left(\omega_{1}\right)}
$$
where $\frac{p\left(\mathbf{x} \mid \omega_{1}\right)}{p\left(\mathbf{x} \mid \omega_{2}\right)}$ is the **likelihood ratio** and $\frac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}} \cdot \frac{P\left(\omega_{2}\right)}{P\left(\omega_{1}\right)}$ is a **constant** independent of $\mathbf {x}$

##### Example 2: Minimum-Error-Rate Classification

Classification setting

- $\Omega=\left\{\omega_{1}, \omega_{2}, \ldots, \omega_{c}\right\}$: $c$ possible states of nature
- $\mathcal{A}=\left\{\alpha_{1}, \alpha_{2}, \ldots, \alpha_{c}\right\}$:  $\alpha_{i}=\operatorname{decide} \omega_{i}, 1 \leq i \leq c$

Zero-one (symmetrical) loss function
$$
\lambda\left(\alpha_{i} \mid \omega_{j}\right)=\left\{\begin{array}{ll}
0 & i=j \\
1 & i \neq j
\end{array} \quad 1 \leq i, j \leq c\right.
$$

- Assign no loss (e.g. 0) to a correct decision
- Assign a unit loss (e.g. 1) to any incorrect decision (**equal cost**)

Proof
$$
\begin{aligned}
R\left(\alpha_{i} \mid \mathbf{x}\right) &=\sum_{j=1}^{c} \lambda\left(\alpha_{i} \mid \omega_{j}\right) \cdot P\left(\omega_{j} \mid \mathbf{x}\right) \\
&=\sum_{j \neq i} \lambda\left(\alpha_{i} \mid \omega_{j}\right) \cdot P\left(\omega_{j} \mid \mathbf{x}\right)+\lambda\left(\alpha_{i} \mid \omega_{i}\right) \cdot P\left(\omega_{i} \mid \mathbf{x}\right) \\
&=\sum_{j \neq i} P\left(\omega_{j} \mid \mathbf{x}\right) \\
&=1-P\left(\omega_{i} \mid \mathbf{x}\right)
\end{aligned}
$$
$1-P\left(\omega_{i} \mid \mathbf{x}\right)$ is the **error rate**, the probability that action $\alpha_i$ is wrong

To **minimum error rate**, decide $\omega_{i}$ if $P\left(\omega_{i} \mid \mathbf{x}\right)>P\left(\omega_{j} \mid \mathbf{x}\right)$ for all $j \neq i$

### Minimax Criterion (最小最大化准则)

Generally, we assume that the **prior probabilities** over the states of nature $\Omega=\left\{\omega_{1}, \omega_{2}, \ldots, \omega_{c}\right\}$ **are fixed**. 

Nonetheless, in some cases we need to design classifiers which can perform well under **varying prior probabilities**.

e.g. the prior probabilities of catching a sea bass or salmon fish might **vary in different regions**

The **minimax criterion** aims to find the classifier which can **minimize the worst overall risk** for **any value of the priors**

#### Example of Two-category classification

Suppose the two-category classifier $\alpha(\cdot)$ decides the feature of $\omega_{1}$ in region $\mathcal{R}_{1}$ and decides the feature of $\omega_{2}$ in region $\mathcal{R}_{2} .$ Here, $\mathcal{R}_{1} \cup \mathcal{R}_{2}=\mathbf{R}^{d}$ and $\mathcal{R}_{1} \cap \mathcal{R}_{2}=\emptyset$
$$
\begin{aligned}
R 
&=\int R(\alpha(\mathbf{x}) \mid \mathbf{x}) \cdot p(\mathbf{x}) d \mathbf{x} \\
&=\int_{\mathcal{R}_{1}} R(\alpha_{1} \mid \mathbf{x}) \cdot p(\mathbf{x}) d \mathbf{x}+\int_{\mathcal{R}_{2}} R(\alpha_{2} \mid \mathbf{x}) \cdot p(\mathbf{x}) d \mathbf{x}
\end{aligned}
$$

where
$$
\begin{aligned}
\int_{\mathcal{R}_{1}} R(\alpha_{1} \mid \mathbf{x}) \cdot p(\mathbf{x}) d \mathbf{x}
& = \int_{\mathcal{R}_{1}} \sum_{j=1}^{2} R\left(\alpha_{1} \mid \omega_{j}\right) \cdot P\left(\omega_{j} \mid \mathbf{x}\right) \cdot p(\mathbf{x}) d \mathbf{x} \\
& = \int_{\mathcal{R}_{1}} \sum_{j=1}^{2} \lambda_{1 j} \cdot P\left(\omega_{j}\right) \cdot p\left(\mathbf{x} \mid \omega_{j}\right) d \mathbf{x} \\
& = \int_{\mathcal{R}_{1}}\left[\lambda_{11} \cdot P\left(\omega_{1}\right) \cdot p\left(\mathbf{x} \mid \omega_{1}\right)+\lambda_{12} \cdot P\left(\omega_{2}\right) \cdot p\left(\mathbf{x} \mid \omega_{2}\right)\right] d \mathbf{x}
\end{aligned}
$$

the same
$$
\int_{\mathcal{R}_{2}} R(\alpha_{2} \mid \mathbf{x}) \cdot p(\mathbf{x}) d \mathbf{x} = \int_{\mathcal{R}_{2}}\left[\lambda_{21} \cdot P\left(\omega_{1}\right) \cdot p\left(\mathbf{x} \mid \omega_{1}\right)+\lambda_{22} \cdot P\left(\omega_{2}\right) \cdot p\left(\mathbf{x} \mid \omega_{2}\right)\right] d \mathbf{x}
$$
Rewrite the overall risk $R$ as a function of $P\left(\omega_{1}\right)$ via

$$
P\left(\omega_{1}\right)=1-P\left(\omega_{2}\right) \\ 
\int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}
=1-\int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}
$$

[In the second equation,$\int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}$ stands for the ratio of the sample $\mathbf{x}$ (its true state of nature is $w_1$)have been classified as $w_1$ (view as True Positive); while $\int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}$ stands for the ratio of the sample $\mathbf{x}$ (its true state of nature is $w_1$)have been classified as $w_2$ (view as False Positive);]

we get 
$$
\begin{aligned}
R=& \lambda_{22}+\left(\lambda_{12}-\lambda_{22}\right) \int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) d \mathbf{x} \\
&+P\left(\omega_{1}\right)\left[\left(\lambda_{11}-\lambda_{22}\right)+\left(\lambda_{21}-\lambda_{11}\right) \int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}-\left(\lambda_{12}-\lambda_{22}\right) \int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) d \mathbf{x}\right]
\end{aligned}
$$
where $R_{mm}$ stands for minimax risk
$$
\begin{array}
& R_{mm} & =\lambda_{22}+\left(\lambda_{12}-\lambda_{22}\right) \int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) d \mathbf{x} \\
& = \lambda_{11}+\left(\lambda_{21}-\lambda_{11}\right) \int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x} \\
\end{array}
$$
so for minimax solution we required:
$$
\left[\left(\lambda_{11}-\lambda_{22}\right)+\left(\lambda_{21}-\lambda_{11}\right) \int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) d \mathbf{x}-\left(\lambda_{12}-\lambda_{22}\right) \int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) d \mathbf{x}\right] = 0
$$


### Discriminant Function (判别函数)

$$
g_{i}: \mathbf{R}^{d} \rightarrow \mathbf{R} \quad(1 \leq i \leq c)
$$

$$
\text { Decide } \omega_{i} \text { if } g_{i}(\mathbf{x})>g_{j}(\mathbf{x}) \text { for all } j \neq i
$$

- Useful way to represent classifiers

- One function per category

- Various **discriminant functions** may leads to **Identical classification results**
  $f(\cdot)$ is a monotonically increasing function (单调递增函数), then
  $$
  f\left(g_{i}(\mathbf{x})\right) \Longleftrightarrow g_{i}(\mathbf{x})
  $$
  which means they are **equivalent in decision**

<img src="images/image-20210311152450165.png" alt="image-20210311152450165" style="zoom: 50%;" />

##### Type

- Minimum risk
  $$
  g_{i}(\mathbf{x})=-R\left(\alpha_{i} \mid \mathbf{x}\right) \quad(1 \leq i \leq c)
  $$

- Minimum-error-rate
  $$
  g_{i}(\mathbf{x})=P\left(\omega_{i} \mid \mathbf{x}\right) \quad(1 \leq i \leq c)
  $$

### Decision region (决策区域)

From $c$ discriminant functions, get $c$ decision regions
$$
g_{i}(\cdot)(1 \leq i \leq c) \rightarrow \mathcal{R}_{i} \subset \mathbf{R}^{d}(1 \leq i \leq c)
$$

##### Definition

$$
\begin{aligned}
\mathcal{R}_{i} &=\left\{\mathbf{x} \mid \mathbf{x} \in \mathbf{R}^{d}: g_{i}(\mathbf{x})>g_{j}(\mathbf{x}) \forall j \neq i\right\} \\
& \text { where } \mathcal{R}_{i} \cap \mathcal{R}_{j}=\emptyset(i \neq j) \text { and } \bigcup_{i=1}^{c} \mathcal{R}_{i}=\mathbf{R}^{d}
\end{aligned}
$$

#### Decision boundary (决策边界)

Surface in feature space where ties occur among several largest discriminant functions

On decision boundaries, more than one $g_i(\cdot)$ is maximum

<img src="images/image-20210315162159921.png" alt="image-20210315162159921" style="zoom:50%;" />

### Some Probability

#### Expected Value $\mu$

$\sim$ stands for has the distribution

- Discrete
  $$
  \begin{array}{l}
  x \in \mathcal{X}=\left\{x_{1}, x_{2}, \ldots, x_{c}\right\} \\
  x \sim P(\cdot) \\
  \mathcal{E}[x]=\sum_{x \in \mathcal{X}} x \cdot P(x)=\sum_{i=1}^{c} x_{i} \cdot P\left(x_{i}\right)
  \end{array}
  $$

- Continuous
  $$
  \begin{array}{l}
  x \in \mathbf{R} \\
  x \sim p(\cdot) \\
  \mathcal{E}[x]=\int_{-\infty}^{\infty} x \cdot p(x) d x
  \end{array}
  $$

#### Variance $\sigma^2$

$$
\operatorname{Var}[x]=\mathcal{E}\left[(x-\mathcal{E}[x])^{2}\right]
$$

- Discrete
  $$
  \operatorname{Var}[x]=\sum_{i=1}^{c}\left(x_{i}-\mu\right)^{2} \cdot P\left(x_{i}\right)
  $$

- Continuous
  $$
  \operatorname{Var}[x]=\int_{-\infty}^{\infty}(x-\mu)^{2} \cdot p(x) d x
  $$

#### Vector Random Variables

- **joint** pdf
  $$
  \mathbf{x} \sim p(\mathbf{x})=p\left(x_{1}, x_{2}, \ldots, x_{d}\right)
  $$

- **marginal** pdf
  $$
  p\left(\mathbf{x}_{1}\right)=\int p\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right) d \mathbf{x}_{2} \\
  \left(\mathbf{x}_{1} \cap \mathbf{x}_{2}=\emptyset ; \mathbf{x}_{1} \cup \mathbf{x}_{2}=\mathbf{x}\right)
  $$

##### Expected vector

$$
\mathcal{E}\left[x_{i}\right]=\int_{-\infty}^{\infty} x_{i} \cdot p\left(x_{i}\right) d x_{i} \quad(1 \leq i \leq d)
$$

#### Covariance Matrix

**Symmetric** & **Positive semidefinite** 
$$
\boldsymbol{\Sigma}=\left[\sigma_{i j}\right]_{1 \leq i, j \leq d}=\left(\begin{array}{cccc}
\sigma_{11} & \sigma_{12} & \ldots & \sigma_{1 d} \\
\sigma_{21} & \sigma_{22} & \ldots & \sigma_{2 d} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{d 1} & \sigma_{d 2} & \ldots & \sigma_{d d}
\end{array}\right)
$$

$$
\begin{aligned}
\sigma_{i j}=\sigma_{j i} &=\mathcal{E}\left[\left(x_{i}-\mu_{i}\right)\left(x_{j}-\mu_{j}\right)\right]\\
&=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty}\left(x_{i}-\mu_{i}\right)\left(x_{j}-\mu_{j}\right) \cdot p\left(x_{i}, x_{j}\right) d x_{i} d x_{j} \\
\end{aligned}
$$

$$
\sigma_{i i}=\operatorname{Var}\left[x_{i}\right]=\sigma_{i}^{2}
$$

where $p\left(x_{i}, x_{j}\right)$ is **marginal pdf** on a pair of random variables $(x_i,x_j)$, exported from **joint pdf**.

### Gaussian Density in Multivariate Case

$$
p(\mathbf{x})=\frac{1}{(2 \pi)^{d / 2}|\mathbf{\Sigma}|^{1 / 2}} \exp \left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{t} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]
$$

where
$$
\mu_{i}=\mathcal{E}\left[x_{i}\right] \quad \sigma_{i j}=\sigma_{j i}=\mathcal{E}\left[\left(x_{i}-\mu_{i}\right)\left(x_{j}-\mu_{j}\right)\right]
$$

and
$$
\begin{array}{l}
\mathrm{x}=\left(x_{1}, x_{2}, \ldots, x_{d}\right)^{t}: d \text { -dimensional column vector } \\
\mathrm{\mu}=\left(\mu_{1}, \mu_{2}, \ldots, \mu_{d}\right)^{t}: d \text { -dimensional mean vector }
\end{array}
$$

$$
\begin{array}{r}
(\mathbf{x}-\boldsymbol{\mu})^{t}: 1 \times d \text { matrix } \\
\boldsymbol{\Sigma}^{-1}: d \times d \text { matrix } \\
(\mathbf{x}-\boldsymbol{\mu}): d \times 1 \text { matrix } \\
(\mathbf{x}-\boldsymbol{\mu})^{t} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}): scalar \ (1 \times 1 \ matrix)
\end{array}
$$

because $\Sigma^{-1}$ is positive definite

$$
(\mathbf{x}-\boldsymbol{\mu})^{t} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) \geq 0
$$

#### Minimum-error-rate classification

$$
g_{i}(\mathbf{x})=P\left(\omega_{i} \mid \mathbf{x}\right)
$$

which the same as
$$
g_i(\mathbf{x})=\ln P(w_i \mid \mathbf{x}) = \ln p(\mathbf{x} \mid w_i) + \ln P(w_i)
$$
assume that
$$
p\left(\mathbf{x} \mid \omega_{i}\right) \sim N\left(\boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)
$$

so we have

$$
g_{i}(\mathbf{x})=-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}_{i}\right)^{t} \boldsymbol{\Sigma}_{i}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_{i}\right)-\frac{d}{2} \ln 2 \pi-\frac{1}{2} \ln \left|\boldsymbol{\Sigma}_{i}\right|+\ln P\left(\omega_{i}\right)
$$
where $\frac{d}{2}\ln 2\pi$ is a constant that could be ignored

##### Case 1: $\boldsymbol{\Sigma}_{i}=\sigma^{2} \mathbf{I}$

$$
\boldsymbol{\Sigma}_{i}=\sigma^{2} \cdot\left(\begin{array}{lllll}
1 & & & \\
& 1 & & \\
& & \ddots & \\
& & & 1
\end{array}\right)=\left(\begin{array}{llll}
\sigma^{2} & & & \\
& \sigma^{2} & & \\
& & \ddots & \\
& & & \sigma^{2}
\end{array}\right)
$$

so we get
$$
g_{i}(\mathbf{x})=-\frac{\left\|\mathbf{x}-\boldsymbol{\mu}_{i}\right\|^{2}}{2 \sigma^{2}}+\ln P\left(\omega_{i}\right)=-\frac{(\mathbf{x}-\boldsymbol{\mu_i})^t(\mathbf{x}-\boldsymbol{\mu_i})}{2 \sigma^{2}}+\ln P\left(\omega_{i}\right)
$$
rearrange
$$
g_{i}(\mathbf{x})=-\frac{1}{2 \sigma^{2}}\left[\mathbf{x}^{t} \mathbf{x}-2 \boldsymbol{\mu}_{i}^{t} \mathbf{x}+\boldsymbol{\mu}_{i}^{t} \boldsymbol{\mu}_{i}\right]+\ln P\left(\omega_{i}\right)
$$
where $\mathbf{x}^{t} \mathbf{x}$ is the same for all states of nature which could be ignored

Finally we get a **Linear discriminant functions** (线性判别函数)
$$
g_{i}(\mathbf{x})=\mathbf{w}_{i}^{t} \mathbf{x}+w_{i 0}
$$
where $\mathbf{w}_{i}=\frac{1}{\sigma^{2}} \boldsymbol{\mu}_{i}$ is weight vector and $w_{i 0}=-\frac{1}{2 \sigma^{2}} \boldsymbol{\mu}_{i}^{t} \boldsymbol{\mu}_{i}+\ln P\left(\omega_{i}\right)$ is threshold/bias 

##### Case 2: $\boldsymbol{\Sigma}_{i}=\boldsymbol{\Sigma}$

$\left(\mathbf{x}-\boldsymbol{\mu}_{i}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_{i}\right)$ is call squared **Mahalanobis distance** (马氏距离)

when $\boldsymbol{\Sigma} = \mathbf{I}$ it reduces to Euclidean distance
$$
g_{i}(\mathbf{x})=-\frac{1}{2}\left[\mathbf{x}^{t} \mathbf{\Sigma}^{-1} \mathbf{x}-2 \boldsymbol{\mu}_{i}^{t} \mathbf{\Sigma}^{-1} \mathbf{x}+\boldsymbol{\mu}_{i}^{t} \mathbf{\Sigma}^{-1} \boldsymbol{\mu}_{i}\right]+\ln P\left(\omega_{i}\right)
$$
where $\mathbf{x}^{t} \mathbf{\Sigma}^{-1} \mathbf{x}$ is the same for all states of nature which could be ignored

Finally we get a **Linear discriminant functions**
$$
g_{i}(\mathbf{x})=\mathbf{w}_{i}^{t} \mathbf{x}+w_{i 0}
$$
where $\mathbf{w}_{i}=\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{i}$ is weight vector and  $w_{i 0}=-\frac{1}{2} \boldsymbol{\mu}_{i}^{t} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{i}+\ln P\left(\omega_{i}\right)$is threshold/bias 

##### Case 3: $\boldsymbol{\Sigma}_{i}=Arbitrary (任意的)$

Using  **quadratic discriminant function** (二次判别函数)
$$
g_{i}(\mathbf{x})=\mathbf{x}^{t} \mathbf{W}_{i} \mathbf{x}+\mathbf{w}_{i}^{t} \mathbf{x}+w_{i 0}
$$
where
$$
\mathbf{W}_{i}=-\frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1} \text{ quadratic matrix} \\
\mathbf{w}_{i}=\boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i} \text { weight vector } \\
w_{i 0}=-\frac{1}{2} \boldsymbol{\mu}_{i}^{t} \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}-\frac{1}{2} \ln \left|\boldsymbol{\Sigma}_{i}\right|+\ln P\left(\omega_{i}\right) \text { threshold/bias }
$$
