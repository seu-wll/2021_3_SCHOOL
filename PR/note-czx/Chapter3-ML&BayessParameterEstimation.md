# Chapter 3 Maximum-Likelihood and Bayesian Parameter Estimation
### Bayes Theorem for Classification

$$
P\left(\omega_{j} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \omega_{j}\right) \cdot P\left(\omega_{j}\right)}{p(\mathbf{x})}(1 \leq j \leq c)
$$

To compute posterior probability $P(w_j \mid \mathbf{x})$, we need to know

- Prior probability $P(w_j)$
- Likelihood $p(\mathbf{x} \mid w_j)$

#### Collection of Training Examples

$$
\mathcal{D}_{j}(1 \leq j \leq c)
$$

- Composed of $c$ data sets
- Each example in $\mathcal{D}_{j}$ is drawn according to the class-conditional pdf $p\left(\mathbf{x} \mid \omega_{j}\right)$

- Examples in $\mathcal{D}_{j}$ are i.i.d. random variables

#### To get prior probability

$$
P\left(\omega_{j}\right)=\frac{\left|\mathcal{D}_{j}\right|}{\sum_{i=1}^{c}\left|\mathcal{D}_{i}\right|}
$$

Here $\mid \cdot \mid $ returns the **cardinality** (集合的势) i.e. **number of elements of a set**

#### To get class-conditional pdf

##### Case 1: $p(\mathbf{x} \mid w_j)$ has certain parametric form

e.g. 

$$
p\left(\mathbf{x} \mid \omega_{j}\right) \sim N\left(\boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)
$$

the parameters are
$$
\boldsymbol{\theta}_{j}=\left\{\boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right\}
$$
we know that $\mathbf{x} \in R^d$ so $\boldsymbol{\theta}_{j}$ contains $d+\frac{d(d+1)}{2}$ free parameters (corresponding to $\boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}$)

To show the dependence, we also write $p(\mathbf{x} \mid w_j)$ as $p(\mathbf{x} \mid w_j, \boldsymbol{\theta})$

##### Case 2: $p(\mathbf{x} \mid w_j)$ has no parametric form

Note that it **doesn’t mean no parameters**

### Estimation Under Parametric Form

#### Maximum-Likelihood (ML) estimation

- View parameters as quantities whose values are **fixed but unknown**
- Estimate parameter values by **maximizing the likelihood** (probability) of observing the actual training examples

#### Bayesian estimation

- View parameters as **random variables** having some known prior distribution

- Observation of the actual training examples transforms parameters’ **prior distribution into posterior distribution** (via Bayes theorem)

### Maximum-Likelihood Estimation

#### Task

Estimate $\left\{\boldsymbol{\theta}_{j}\right\}_{j=1}^{c}$ from $\left\{\mathcal{D}_{j}\right\}_{j=1}^{c}$

#### A simplified treatment

Examples in $\mathcal{D}_{j}$ gives no information about $\boldsymbol{\theta}_{i}$ if $i \neq j$

**Work with each category separately** and therefore simplify the notations by dropping subscripts w.r.t. categories $\mathcal{D}_{j} \rightarrow \mathcal{D}; \boldsymbol{\theta}_{j} \rightarrow \boldsymbol{\theta}$

#### Definitions

- $\mathbf{x}_{k} \sim p(\mathbf{x} \mid \boldsymbol{\theta}) \quad (k=1,\dots,n)$
- $\boldsymbol{\theta}:$ Parameters to be estimated
- $\mathcal{D}:$ A set of i.i.d. examples $\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{n}\right\}$

#### The objective function

$$
p(\mathcal{D} \mid \boldsymbol{\theta})=\prod_{k=1}^{n} p\left(\boldsymbol{x}_{k} \mid \boldsymbol{\theta}\right)
$$

The likelihood of $\boldsymbol{\theta}$ w.r.t. the set of observed examples

The goal is
$$
\hat{\boldsymbol{\theta}}=\arg \max _{\boldsymbol{\theta}} p(\mathcal{D} \mid \boldsymbol{\theta})
$$

Define **log-likelihood function**
$$
l(\boldsymbol{\theta})=\ln p(\mathcal{D} \mid \boldsymbol{\theta})
$$
so the goal could be rewrite as
$$
\hat{\boldsymbol{\theta}}=\arg \max _{\boldsymbol{\theta}} l(\boldsymbol{\theta})
$$
the necessary conditions for ML estimate $\hat{\boldsymbol{\theta}}$
$$
\nabla_{\boldsymbol{\theta}} l_{\left.\right|_{\theta=\hat{\theta}}}=\mathbf{0}
$$

#### The Gaussian Case

$$
\mathbf{x}_{k} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

##### Case 1: $\Sigma$ is known

For each component

$$
p\left(\mathbf{x}_{k} \mid \boldsymbol{\mu}\right)=\frac{1}{(2 \pi)^{d / 2}|\mathbf{\Sigma}|^{1 / 2}} \exp \left[-\frac{1}{2}\left(\mathbf{x}_{k}-\boldsymbol{\mu}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{k}-\boldsymbol{\mu}\right)\right]
$$

$$
\begin{array}{l}
&\ln p\left(\mathbf{x}_{k} \mid \boldsymbol{\mu}\right)&=-\frac{1}{2} \ln \left[(2 \pi)^{d}|\boldsymbol{\Sigma}|\right]-\frac{1}{2}\left(\mathbf{x}_{k}-\boldsymbol{\mu}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{k}-\boldsymbol{\mu}\right) \\
&&=-\frac{1}{2} \ln \left[(2 \pi)^{d}|\boldsymbol{\Sigma}|\right]-\frac{1}{2} \mathbf{x}_{k}^{t} \boldsymbol{\Sigma}^{-1} \mathbf{x}_{k}+\boldsymbol{\mu}^{t} \boldsymbol{\Sigma}^{-1} \mathbf{x}_{k}-\frac{1}{2} \boldsymbol{\mu}^{t} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}
\end{array}
$$

$$
\boldsymbol{\nabla}_{\boldsymbol{\mu}} \ln p\left(\mathbf{x}_{k} \mid \boldsymbol{\mu}\right)=\boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{k}-\boldsymbol{\mu}\right)
$$

so we have
$$
\nabla_{\mu} l=\sum_{k=1} \Sigma^{-1}\left(\mathbf{x}_{k}-\mu\right) = 0
$$
then we multiply $\Sigma$ on both sizes
$$
\sum_{k=1}^{n}\left(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}\right)=\mathbf{0}
$$
that is 
$$
\hat{\boldsymbol{\mu}}=\frac{1}{n} \sum_{k=1}^{n} \mathbf{x}_{k}
$$
which is a intuitive result

#### Case 2: $\Sigma$ is unknown

Firstly we consider univariate case:
$$
p\left(x_{k} \mid \boldsymbol{\theta}\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left[-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right]
$$
where
$$
\boldsymbol{\theta}=\left[\begin{array}{l}
\theta_{1} \\
\theta_{2}
\end{array}\right]=\left[\begin{array}{c}
\mu \\
\sigma^{2}
\end{array}\right]
$$

use $\ln$ function

$$
\ln p\left(x_{k} \mid \boldsymbol{\theta}\right)=-\frac{1}{2} \ln 2 \pi \theta_{2}-\frac{1}{2 \theta_{2}}\left(x_{k}-\theta_{1}\right)^{2}
$$
so we get
$$
\boldsymbol{\nabla}_{\boldsymbol{\theta}} \ln l(\boldsymbol{\hat{\theta}})=\boldsymbol{\nabla}_{\boldsymbol{\theta}} \ln p\left(x_{k} \mid \boldsymbol{\theta}\right)=\left[
\begin{array}{c}
\frac{1}{\theta_{2}}\left(x_{k}-\theta_{1}\right) \\
-\frac{1}{2 \theta_{2}}+\frac{\left(x_{k}-\theta_{1}\right)^{2}}{2 \theta_{2}^{2}}
\end{array}
\right]=\left[
\begin{array}{c}
0 \\
0 \\
\end{array}
\right]
$$
that is
$$
\hat{\theta}_{1}=\frac{1}{n} \sum_{k=1}^{n} x_{k}
$$

$$
\hat{\theta}_{2}=\frac{1}{n} \sum_{k=1}^{n}\left(x_{k}-\hat{\theta}_{1}\right)^{2}
$$

in multivariate case
$$
\hat{\boldsymbol{\mu}}=\frac{1}{n} \sum_{k=1}^{n} \mathbf{x}_{k}
$$

$$
\hat{\boldsymbol{\Sigma}}=\frac{1}{n} \sum_{k=1}^{n}\left(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}\right)\left(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}\right)^{t}
$$

### Bayesian Estimation

- The **parametric form** of the likelihood function for each category is known $p\left(\mathbf{x} \mid \omega_{j}, \boldsymbol{\theta}_{j}\right)(1 \leq j \leq c)$

- Consider $\boldsymbol{\theta}$ as **random variables**

- Fully exploit training examples
  $$
  P\left(\omega_{j} \mid \mathbf{x}, \mathcal{D}^{*}\right)
  $$
  
  $$
  \left(\mathcal{D}^{*}=\mathcal{D}_{1} \cup \mathcal{D}_{2} \cup \cdots \cup \mathcal{D}_{c}\right)
  $$

#### Analyze

$$
P\left(\omega_{j} \mid \mathbf{x}, \mathcal{D}^{*}\right)=\frac{p\left(\omega_{j}, \mathbf{x}, \mathcal{D}^{*}\right)}{p\left(\mathbf{x}, \mathcal{D}^{*}\right)}=\frac{p\left(\omega_{j}, \mathbf{x}, \mathcal{D}^{*}\right)}{\sum_{i=1}^{c} p\left(\omega_{i}, \mathbf{x}, \mathcal{D}^{*}\right)}
$$

where
$$
p\left(\omega_{j}, \mathbf{x}, \mathcal{D}^{*}\right)=p\left(\mathcal{D}^{*}\right) \cdot p\left(\omega_{j}, \mathbf{x} \mid \mathcal{D}^{*}\right)=p\left(\mathcal{D}^{*}\right) \cdot P\left(\omega_{j} \mid \mathcal{D}^{*}\right) \cdot p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}^{*}\right)
$$
so
$$
\begin{array}
&P\left(\omega_{j} \mid \mathbf{x}, \mathcal{D}^{*}\right)
&=\frac{p\left(\mathcal{D}^{*}\right) \cdot P\left(\omega_{j} \mid \mathcal{D}^{*}\right) \cdot p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}^{*}\right)}{p\left(\mathcal{D}^{*}\right) \cdot \sum_{i=1}^{c} P\left(\omega_{i} \mid \mathcal{D}^{*}\right) \cdot p\left(\mathbf{x} \mid \omega_{i}, \mathcal{D}^{*}\right)} \\
&=\frac{P\left(\omega_{j} \mid \mathcal{D}^{*}\right) \cdot p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}^{*}\right)}{\sum_{i=1}^{c} P\left(\omega_{i} \mid \mathcal{D}^{*}\right) \cdot p\left(\mathbf{x} \mid \omega_{i}, \mathcal{D}^{*}\right)}
\end{array}
$$
with two assumptions
$$
\begin{array}{l}
P\left(\omega_{j} \mid \mathcal{D}^{*}\right)=P\left(\omega_{j}\right) \\
p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}^{*}\right)=p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}_{j}\right)
\end{array}
$$
(in first equation, Prior Probability is independent of Dataset)

we have
$$
P\left(\omega_{j} \mid \mathbf{x}, \mathcal{D}^{*}\right)=\frac{P\left(\omega_{j}\right) \cdot p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}_{j}\right)}{\sum_{i=1}^{c} P\left(\omega_{i}\right) \cdot p\left(\mathbf{x} \mid \omega_{i}, \mathcal{D}_{i}\right)}
$$
to calculate, the key problem is to determine $p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}_{j}\right)$

since we treat each class independently, we simplify the class-conditional pdf notation
$$
p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}_{j}\right) \rightarrow p(\mathbf{x} \mid \mathcal{D})
$$

introducing $\boldsymbol{\theta}$ which is a random variable w.r.t. parametric form

$$
\begin{array}
&p(\mathbf{x} \mid \mathcal{D})
&=\int p(\mathbf{x}, \boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta}\\
&=\int p(\mathbf{x} \mid \boldsymbol{\theta}, \mathcal{D}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta}\\
&=\int p(\mathbf{x} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta}\\
\end{array}
$$
where $\mathrm{x}$ is independent of $\mathcal{D}$ given $\boldsymbol{\theta}$

#### General Procedure

##### Phase 1: prior pdf $\rightarrow$ posterior pdf (for $\boldsymbol{\theta}$)

<img src="images/image-20210325194459243.png" alt="image-20210325194459243" style="zoom:50%;" />

where
$$
\begin{aligned}
p(\boldsymbol{\theta} \mid \mathcal{D}) &=\frac{p(\boldsymbol{\theta}, \mathcal{D})}{p(\mathcal{D})} \\
&=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{\int p(\boldsymbol{\theta}, \mathcal{D}) d \boldsymbol{\theta}} \\
&=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{\int p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta}) d \boldsymbol{\theta}}
\end{aligned}
$$

$$
p(\mathcal{D} \mid \boldsymbol{\theta})=\prod_{k=1}^{n} p\left(\mathbf{x}_{k} \mid \boldsymbol{\theta}\right)
$$

##### Phase 2: posterior pdf (for $\boldsymbol{\theta}$) $\rightarrow$ class-conditional pdf (for $\mathbf{x}$)

<img src="images/image-20210325194746084.png" alt="image-20210325194746084" style="zoom:50%;" />
$$
p(\mathbf{x} \mid \mathcal{D})=\int p(\mathbf{x} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta}
$$

##### Phase 3: posterior pdf (for $\mathbf{x}, \boldsymbol{D^*}$)

$$
P\left(\omega_{j} \mid \mathbf{x}, \mathcal{D}^{*}\right)=\frac{P\left(\omega_{j}\right) \cdot p\left(\mathbf{x} \mid \omega_{j}, \mathcal{D}_{j}\right)}{\sum_{i=1}^{c} P\left(\omega_{i}\right) \cdot p\left(\mathbf{x} \mid \omega_{i}, \mathcal{D}_{i}\right)}
$$

#### Example 1: Gaussian Case & Unknown $\boldsymbol{\mu}$

we assume
$$
\begin{array}{l}
p(x \mid \mu) \sim N\left(\mu, \sigma^{2}\right) \\
p(\mu) \sim N\left(\mu_{0}, \sigma_{0}^{2}\right)
\end{array}
$$
Note that for $p(\mu)$ here, we could also assume other form of prior pdf;
$$
\begin{aligned}
p(\mu \mid \mathcal{D}) =\frac{p(\mu, \mathcal{D})}{p(\mathcal{D})} 
&=\frac{p(\mu) p(\mathcal{D} \mid \mu)}{\int p(\mu) p(\mathcal{D} \mid \mu) d \mu} \\
&=\alpha p(\mu) p(\mathcal{D} \mid \mu) \\
&=\alpha p(\mu) \prod_{k=1}^{n} p\left(x_{k} \mid \mu\right)
\end{aligned}
$$
where $\int p(\mu) p(\mathcal{D} \mid \mu) d \mu$ is a constant not related to $\mu$, rewrite as $\alpha$; examples in $\mathcal{D}$ are i.i.d

continue
$$
\begin{array}
&p(\mu \mid \mathcal{D})
&=\alpha p(\mu) \prod_{k=1}^{n} p\left(x_{k} \mid \mu\right)\\
&=\alpha \cdot \frac{1}{\sqrt{2 \pi} \sigma_{0}} \exp \left[-\frac{1}{2}\left(\frac{\mu-\mu_{0}}{\sigma_{0}}\right)^{2}\right] \cdot \prod_{k=1}^{n} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left[-\frac{1}{2}\left(\frac{x_{k}-\mu}{\sigma}\right)^{2}\right]\\
& =\alpha^{\prime} \cdot \exp \left[-\frac{1}{2}\left(\left(\frac{\mu-\mu_{0}}{\sigma_{0}}\right)^{2}+\sum_{k=1}^{n}\left(\frac{\mu-x_{k}}{\sigma}\right)^{2}\right)\right]\\
&=\alpha^{\prime \prime} \cdot \exp \left[-\frac{1}{2}\left[\left(\frac{n}{\sigma^{2}}+\frac{1}{\sigma_{0}^{2}}\right) \mu^{2}-2\left(\frac{1}{\sigma^{2}} \sum_{k=1}^{n} x_{k}+\frac{\mu_{0}}{\sigma_{0}^{2}}\right) \mu\right]\right]
\end{array}
$$
Note that $p(\mu \mid \mathcal{D})$ is an **exponential function** of a **quadratic function** of $\mu$

So that $p(\mu \mid \mathcal{D})$ is a normal pdf as well

that is 
$$
p(\mu \mid \mathcal{D}) \sim N\left(\mu_{n}, \sigma_{n}^{2}\right)
$$

$$
\begin{aligned}
\sigma_{n}^{2} &=\frac{\sigma^{2} \sigma_{0}^{2}}{n \sigma_{0}^{2}+\sigma^{2}} \\
\mu_{n} &=\frac{\sigma_{n}^{2}}{\sigma^{2}} \sum_{k=1}^{n} x_{k}+\frac{\sigma_{n}^{2}}{\sigma_{0}^{2}} \mu_{0}
\end{aligned}
$$

with the result, considering
$$
\begin{array}
&p(x \mid \mathcal{D})
&=\int p(x \mid \mu) p(\mu \mid \mathcal{D}) d \mu \\
&=\int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right] \frac{1}{\sqrt{2 \pi} \sigma_{n}} \exp \left[-\frac{1}{2}\left(\frac{\mu-\mu_{n}}{\sigma_{n}}\right)^{2}\right] d \mu \\
&=\beta \cdot \exp \left[-\frac{1}{2} \frac{\left(x-\mu_{n}\right)^{2}}{\sigma^{2}+\sigma_{n}^{2}}\right] \cdot \int \text{[pdf function of a Norm]} \\
&=\beta \cdot \exp \left[-\frac{1}{2} \frac{\left(x-\mu_{n}\right)^{2}}{\sigma^{2}+\sigma_{n}^{2}}\right]
\end{array}
$$
Note that $p(x \mid \mathcal{D})$ is an **exponential function** of a **quadratic function** of $x$

So that $p(x \mid \mathcal{D})$ is a normal pdf as well again

that is 
$$
p(x \mid \mathcal{D}) \sim N\left(\mu_{n}, \sigma^{2}+\sigma_{n}^{2}\right)
$$

#### Example 2: Gaussian Case & Multivariate & Unknown $\boldsymbol{\mu}$

$$
\begin{array}{l}
p(\mathbf{x} \mid \boldsymbol{\mu}) \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \\
p(\boldsymbol{\mu}) \sim N\left(\boldsymbol{\mu}_{0}, \boldsymbol{\Sigma}_{0}\right)
\end{array}
$$

so that
$$
p(\boldsymbol{\mu} \mid \mathcal{D}) \sim N\left(\boldsymbol{\mu}_{n}, \boldsymbol{\Sigma}_{n}\right) \\
p(\mathbf{x} \mid \mathcal{D}) \sim N\left(\boldsymbol{\mu}_{n}, \boldsymbol{\Sigma}+\boldsymbol{\Sigma}_{n}\right)
$$
where
$$
\mu_{n}=\Sigma_{0}\left(\Sigma_{0}+\frac{1}{n} \Sigma\right)^{-1} \frac{1}{n} \sum_{k=1}^{n} \mathbf{x}_{k}+\frac{1}{n} \Sigma\left(\Sigma_{0}+\frac{1}{n} \Sigma\right)^{-1} \mu_{0} \\
\Sigma_{n}=\Sigma_{0}\left(\Sigma_{0}+\frac{1}{n} \Sigma\right)^{-1} \frac{1}{n} \Sigma
$$

### Comparing ML estimation & Bayes estimation

<img src="images/image-20210325202422649.png" alt="image-20210325202422649" style="zoom:67%;" />

#### The source of classification error

$$
\text {Bayes error} \ + \ \text {Model error} \ + \ \text {Estimation error}
$$

