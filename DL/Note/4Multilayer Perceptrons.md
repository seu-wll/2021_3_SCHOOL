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

