# 4.1. Multilayer Perceptrons

## 4.1.1. Hidden Layers



 H is also known as a *hidden-layer variable* or a *hidden variable*



Still linear model since an affine function of an affine function is itself an affine function.

nonlinear *activation function* σ



## 4.1.2. Activation Functions

### ReLU Function

$$ {\operatorname{ReLU}(x) = \max(x, 0).}
\operatorname{ReLU}(x) = \max(x, 0).
$$

This makes optimization better behaved and it mitigated the well-documented problem of vanishing gradients that plagued previous versions of neural networks 

==Why ReLU can solve the vanishing gradients problem?==

```
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

```

```
y.backward(torch.ones_like(x), retain_graph=True)
```



### Sigmoid Function

smooth 

 mostly been replaced by the simpler and more easily trainable ReLU for most use in hidden layers 

### Tanh Function

About origin symmetry 



# 4.2. Implementation of Multilayer Perceptrons from Scratch



the hidden layer number is 256 . and choose layer widths in power of  2 because it will be computationally efficient due to the mechanism of memory allocation.



if you want use gradient descent, it must be true

```
requires_grad=True
```



torch.reshape():

A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in `input`.

# 4.4. Model Selection, Underfitting, and Overfitting

be sure that we have truly discovered a *general* pattern and not simply memorized our data



**?? When working with finite samples, we run the risk that we might discover apparent associations that turn out not to hold up when we collect more data**



[数据集的划分--训练集、验证集和测试集](https://blog.csdn.net/qq_43741312/article/details/96994243)

验证集是为了选择最优的超参数（网络，学习率），训练集训练普通参数。![img](https://img-blog.csdnimg.cn/20190724110042475.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNzQxMzEy,size_16,color_FFFFFF,t_70)



### 4.4.4.1. Generating the Dataset

 **？？For optimization, we typically want to avoid very large values of gradients or losses. This is why the *features* are rescaled** 



# 4.5. Weight Decay

*Weight decay* (commonly called L2L2 regularization)

函数和0之间的距离越小越简单

L2-regularized：*ridge regression* algorithm

L1L1-regularized：

# 4.8. Numerical Stability and Initialization

[有关梯度消失与梯度爆炸](https://www.jianshu.com/p/3f35e555d5ba)