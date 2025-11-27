# README of Experiment 4 
Replication of Kernels (Can our method replicate existing kernels?) 

<u>**Note**</u>: In this experiment we will assume in our network (the composite small kernel) are all of the same functional form (of the target kernel)(e.g. if we want to mimic an RBF, we create these small kernels all rbf) we want to fit the parameters of the kernel. In future experiments we can try to learn the funcitnal form of each sub-kernel as well.

Specifically, we will try and replicate  
## (1) Linear Kernel 
Choosing a $\sigma^2$ and then fitting our model to replicate $K(x_1, x_2)= \sigma^2 x_1^T x_2$, where $x_1,x_2 \in \mathbb{R}^p$ (i.e. $x_1,x_2$ are column vecors of dimension $p \times 1$).
## (2) RBF Kernel 
Choosing a $\sigma^2$ and lengthscale $l$ and then fitting our model to replicate $K(x_1, x_2)= \sigma^2 \exp(-\frac{||x_1 - x_2||^2}{2l^2})$, where $x_1,x_2 \in \mathbb{R}^p$ (i.e. $x_1,x_2$ are column vecors of dimension $p \times 1$).
## (3) Periodic Kernel 
## (4) Matern Kernel 
## (5) Composition of Kernels (Sum and Product of the above kernels) 
  ### (5.a) 
