# 5.1

#### 在MySequential(nn.Module)的类的定义中，forward在哪里有所表示？x=block（x）的意思是什么？



#### 为什么只需要确定forward就行而不用backwords？

backwords可以自动更新，但是forward更加像是在这一层需要做什么事情。



#### modules的作用是同步更新参数？（怎么理解）



### 5.1.3

```python
def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及`relu`和`dot`函数。
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.linear(X)
```

#### 全连接层共享参数是什么意思？

在代码的上下删除linear的线性层也没有问题。





# 5.2



# 5.4

在执行

```
dense(torch.rand(2, 5))
```

的时候，为什么会自动执行计算，中间调用了什么函数呢？

自动调用forward，本身每一层做的事情就是一个函数，然后会进行计算。



