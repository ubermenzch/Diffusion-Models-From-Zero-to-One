# Diffusion-Models-From-Zero-to-One

从零理解扩散模型：从一个像素通道的加噪公式开始，逐步理解扩散模型如何学习去噪，并最终从随机噪声生成图像。

# Diffusion

加噪过程
假设我们有一张彩色图，其中某个像素的某个颜色通道的值为$`x_0`$。则加噪过程如下

```math
x_1
=
\sqrt{1-\beta_1}x_0
+
\sqrt{\beta_1}\epsilon_1
```

其中$`\epsilon_1`$是噪声值，采样于标准高斯变量$`\varepsilon\sim\mathcal N(0,1)`$，$`\beta_1`$是人为设定的第1步加噪的噪声强度，一般要求$`0<\beta<1`$。写成随机变量形式则有

```math
X_1
=
\sqrt{1-\beta_1}x_0
+
\sqrt{\beta_1}\varepsilon_1
```

加噪过程的通式为

```math
X_t
=
\sqrt{1-\beta_t}x_{t-1}
+
\sqrt{\beta_t}\varepsilon_t,\quad \varepsilon_t \sim \mathcal N(0,1)
```

严格来讲$`X_t`$服从条件高斯分布

```math
\mathcal L\left(X_t\mid X_{t-1}=x_{t-1},X_0=x_0\right)
=\mathcal N
\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_t
\right)
```

由于$`X_t`$仅依赖上个时刻的结果，故加噪过程本质是一个马尔可夫过程，则可以将$`X_0=x_0`$的条件省略

```math
\mathcal{L}\left(X_t\mid X_{t-1}=x_{t-1}\right)
=
\mathcal N
\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_t
\right)
```

可以进一步简写为

```math
\mathcal L (X_t\mid x_{t-1})
=
\mathcal N
\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_t
\right)
```

我们用$`q_t`$表示条件概率密度值，则有

```math
q_t(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t
\right)
```

设

```math
\begin{gathered}
\alpha_t\doteq 1-\beta_t,\quad 1>\alpha_t>0
\\\bar{\alpha}_t \doteq \prod_{i=1}^t \alpha_i,\quad 1>\bar\alpha_t>0
\end{gathered}
```

假如我们不指定$`x_{t-1}`$，则递推式可以写成

```math
X_t
=
\sqrt{\alpha_t}X_{t-1}
+
\sqrt{1-\alpha_t}\varepsilon_t
```

则在给定$`X_0=x_0`$的条件下，有

```math
\begin{aligned}
X_t
&= \sqrt{\alpha_t} X_{t-1} + \sqrt{1 - \alpha_t} \varepsilon_t \\
&= \sqrt{\alpha_t}
\left(
\sqrt{\alpha_{t-1}} X_{t-2}
+
\sqrt{1 - \alpha_{t-1}} \varepsilon_{t-1}
\right)
+
\sqrt{1 - \alpha_t} \varepsilon_t \\
&= \sqrt{\alpha_t \alpha_{t-1}} X_{t-2}
+
\sqrt{\alpha_t (1 - \alpha_{t-1})} \varepsilon_{t-1}
+
\sqrt{1 - \alpha_t} \varepsilon_t \\
&\vdots \\
&= \sqrt{\bar{\alpha}_t} x_0
+
\sum_{i=1}^{t-1}
\left(
\sqrt{\prod_{j=i+1}^{t} \alpha_j}
\sqrt{1 - \alpha_i}
\varepsilon_i
\right)
+
\sqrt{1 - \alpha_t} \varepsilon_t
\\&=\sqrt{\bar{\alpha}_t} x_0
+
\sum_{i=1}^{t}
\left(
\sqrt{\prod_{j=i+1}^{t} \alpha_j}
\sqrt{1 - \alpha_i}
\varepsilon_i
\right),\quad \left(\text{约定空乘积 }\sqrt{\prod_{j=t+1}^{t}\alpha_j}=1\right)
\end{aligned}
```

可以看出，在给定$`X_0=x_0`$的条件下，随机变量$`X_t`$等价于由多个互相独立的高斯随机变量$`\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_t`$表示的线性组合。因为独立高斯分布的线性组合依然是高斯分布，因此在给定$`X_0=x_0`$的条件下，$`X_t`$仍然服从高斯分布。对于任意两个随机变量X和Y，有$`\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y)`$，则对任意$`i\neq j`$都有$`\mathrm{Var}(\varepsilon_i + \varepsilon_j) = \mathrm{Var}(\varepsilon_i) + \mathrm{Var}(\varepsilon_j) + 2\mathrm{Cov}(\varepsilon_i, \varepsilon_j)=\mathrm{Var}(\varepsilon_i) + \mathrm{Var}(\varepsilon_j)`$。则有

```math
\begin{aligned}
\mathrm{Var}
\left(
X_t\mid x_0
\right)
&=
\sum_{i=1}^{t-1}
\mathrm{Var}
\left(
\sqrt{\prod_{j=i+1}^{t}\alpha_j}
\sqrt{1-\alpha_i}
\varepsilon_i
\right)
+
\mathrm{Var}
\left(
\sqrt{1-\alpha_t}\varepsilon_t
\right)
\\
&=
\sum_{i=1}^{t-1}
\left(
\prod_{j=i+1}^{t}\alpha_j
\right)
(1-\alpha_i)
+
(1-\alpha_t)
\\
&=
\sum_{i=1}^{t-1}
\left(
\prod_{j=i+1}^{t}\alpha_j
-
\prod_{j=i}^{t}\alpha_j
\right)
+
(1-\alpha_t)
\\
&=
\left(
\prod_{j=2}^{t}\alpha_j
-
\prod_{j=1}^{t}\alpha_j
\right)
+
\left(
\prod_{j=3}^{t}\alpha_j
-
\prod_{j=2}^{t}\alpha_j
\right)
+
\cdots
+
\left(
\prod_{j=t}^{t}\alpha_j
-
\prod_{j=t-1}^{t}\alpha_j
\right)
+
1-\alpha_t
\\
&=
-\prod_{j=1}^{t}\alpha_j
+
\prod_{j=t}^{t}\alpha_j
+
1-\alpha_t
\\
&=
-\bar{\alpha}_t
+
\alpha_t
+
1-\alpha_t
\\
&=
1-\bar{\alpha}_t .
\end{aligned}
```

因此我们可以直接写出从$`x_0`$直接去采样$`X_t`$的条件分布

```math
\mathcal L (X_t\mid x_0)
=
\mathcal N
\left(
\sqrt{\bar{\alpha}_t}x_0,
1-\bar{\alpha}_t
\right)
```

则

```math
X_t= \sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\bar\varepsilon_t,\quad \bar\varepsilon_t\sim \mathcal  N(0,1)
```

其中$`\bar\varepsilon_t`$用于和递推式中的局部噪声$`\varepsilon_t`$做区分，两个随机变量虽然都服从标准正态分布，但两者是不同的随机变量，两者的关系为

```math
\sqrt{1-\bar\alpha_t}\bar\varepsilon_t=\sum_{i=1}^{t}
\left(
\sqrt{\prod_{j=i+1}^{t} \alpha_j}
\sqrt{1 - \alpha_i}
\varepsilon_i
\right),\quad \left(\text{约定空乘积 }\sqrt{\prod_{j=t+1}^{t}\alpha_j}=1\right)
```

条件概率密度函数为
$`q_t(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar \alpha_t}x_{0},(1-\bar \alpha_t))`$
直接采样等价于一步步采样的本质含义不是说直接采样得到$`x_t`$和一步步采样得到$`x_t`$在值上相同，而是说在经过无穷次采样后，两种方式采样得到的$`x_t`$形成的分布相同。当我们选择合适的$`\beta`$序列使得$`\lim_{t\to\infty}\bar \alpha_t=\prod_{i=1}^t \alpha_i=\prod_{i=1}^t (1-\beta_i)=0`$时，则当T足够大时，就有
$`q_T(x_T\mid x_0)\approx\mathcal{N}(x_T;0,1)`$。 
生成过程
当给定$`x_0`$后，后续所有时刻结果的随机变量$`X_1,X_2,\cdots,X_T`$的解析式都可以直接写出。如果我们可以构造一个神经网络$`\theta`$去学习根据$`x_t`$和$`t`$输出分布$`\mathcal L(X_{t-1}\mid x_t)`$，那么我们就可以根据这个分布逆转加噪过程。重复这个逆转过程最终就可以得到$`X_0`$的分布（训练所用的$`x_0`$的数据分布）。这里可能产生疑问：从$`x_t`$到$`x_0`$，这中间$`x_{t-1}`$依赖于$`x_t`$，整个路径都是随机的，我怎么能确保$`x_T`$生成确定的$`x_0`$呢？实际上，现在要介绍的生成过程还没有生成图片的能力，只有生成分布的能力，也就是说无法从$`x_T`$生成确定的$`x_0`$，但可以训练达到从$`x_T`$生成一批$`x_0`$，这批$`x_0`$的分布符合训练时候所使用的$`x_0`$的数据分布。介绍一下联合概率密度：对于两个连续的随机变量X,Y，当两随机变量互相独立时，他们的联合概率密度函数为
$`f_{X,Y}(x,y)=f_X(x)\cdot f_Y(y)`$
当两随机变量不互相独立时，他们的联合概率密度函数为
$`f_{X,Y}(x,y)=f_X(x)\cdot f_{Y\mid X}(y\mid x)=f_Y(y)\cdot f_{X\mid Y}(x\mid y)`$
在给定$`x_0`$后，将后续结果对应的随机变量联合起来

```math
X_{1:T}
\doteq
\begin{pmatrix}
X_1\\
X_2\\
\vdots\\
X_T
\end{pmatrix}
```

也可以称$`X_{1:T}`$为随机向量。$`X,Y`$可以组成联合高斯（多元高斯分布）的充分必要条件是：对所有非零线性组合$`aX+bY`$，结果必须是高斯变量或退化高斯。其中非零线性组合即指$`a,b`$不同时为0；退化高斯指$`\mathcal N(c,0),c\in \mathbb R`$。由于$`a=1,b=0`$或$`a=0,b=1`$的线性组合也必须是高斯变量或退化高斯，因此这也要求$`X,Y`$都分别是高斯变量或退化高斯。同时联合高斯随机变量的条件分布仍然是高斯分布，即$`X\mid Y=y`$也一定服从高斯分布（在$`Y=y`$的条件下$`X`$服从高斯分布）。随机向量服从多元高斯可以记为

```math
\begin{pmatrix}
X\\
Y
\end{pmatrix}
\sim
\mathcal N
\left(
\mu,
\Sigma
\right),\mu=\begin{pmatrix}
\mu_X\\
\mu_Y
\end{pmatrix},\Sigma
=
\begin{pmatrix}
\mathrm{Var}(X) & \mathrm{Cov}(X,Y)\\
\mathrm{Cov}(Y,X) & \mathrm{Var}(Y)
\end{pmatrix}
```

其中$`\mu`$代表均值向量，$`\Sigma`$代表X,Y的协方差矩阵。假设某个生成过程正确一步步生成得到了$`x_0`$，采样值依次为$`x_{T},\cdots,x_{1},x_0`$。我们记生成过程的概率密度为
$`p_{t-1}(x_{t-1}\mid x_t)`$
则得到该采样序列的联合概率密度函数为

```math
\begin{aligned}
p_{0:T} (x_{0:T})&=p_T(x_T) p_{T-1}(x_{T-1} \mid x_T) p_{T-2}(x_{T-2} \mid x_{T}, x_{T-1}) \cdots p_{0}(x_0 \mid x_T, \dots, x_{1})
\\&=p_T(x_T) p_{T-1}(x_{T-1} \mid x_T) p_{T-2}(x_{T-2} \mid  x_{T-1}) \cdots p_{0}(x_0 \mid x_{1})\qquad \text{（由马尔可夫性质）}
\\&= p_T(x_T) \prod_{t=1}^T p_{t-1}(x_{t-1} \mid x_t)
\end{aligned}
```

由于$`X_T`$的先验分布在DDPM标准中固定为标准高斯，即
$`p_T(x_T)=\mathcal{N}(x_T;0,1)`$
要注意，上面的路径$`x_T,x_{T-1},\cdots,x_{0}`$只是从$`x_T`$到$`x_0`$的其中一条路径，除此之外还包含无数条通往$`x_0`$的路径，所以最终生成$`x_0`$的概率密度（因为只是对联合变量中除$`x_0`$以外的随机变量进行了积分，所以积分结果依旧是$`x_0`$的边缘概率密度函数）应为
$`p_0(x_0)=\int p_{0:T}(x_{1:T},x_0) \, dx_{1:T}`$
最直接的训练目标，就是最大化$`p_0(x_0)`$，为了简化运算，一般会转化为最大化$`\log p_0(x_0)`$。但直接优化$`\log p_0(x_0)`$是很困难的，因此我们引入已知的、容易处理的前向过程的分布$`q(x_{1:T}\mid x_0)`$并通过重要性采样来简化上式积分过程

```math
\begin{aligned}
p_0(x_0) &= \int p_{0:T}(x_{0:T}) \, dx_{1:T}
\\&= \int q_{1:T}(x_{1:T} \mid x_0) \cdot \frac{p_{0:T}(x_{1:T},x_0)}{q_{1:T}(x_{1:T} \mid x_0)} \, dx_{1:T}
\\&=\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)} \left[ \frac{p_{0:T}(x_0,X_{1:T})}{q_{1:T}(X_{1:T} \mid x_0)} \right],\quad p_{0:T}(x_0,X_{1:T})=p_T(X_T) \prod_{t=2}^T p_{t-1}(X_{t-1} \mid X_t)p_0(x_{0} \mid X_1)
\end{aligned}
```

此时$`X_1,X_2,\cdots,X_T`$表示服从前向条件分布的联合随机变量。设$`X`$为随机变量，$`f(x)`$为其概率密度函数；若将$`Y\doteq X_1+X_2`$代入概率密度函数$`f(x)`$就得到了另一个随机变量$`f(Y)`$。因此$`p_{0:T}(x_0,X_{1:T}),q_{1:T}(X_{1:T} \mid x_0),p(X_T),p_{t-1}(X_{t-1} \mid X_t),p_0(x_{0} \mid X_1)`$都是随机变量。其中，$`X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)`$表示$`X_1,X_{2},\cdots,X_T`$服从联合条件分布$`Q_{1:T}(\cdot\mid x_0)`$；$`X_{1:T}`$中每个随机变量同时也都服从一个边缘条件分布$`Q_t(\cdot\mid x_0)`$。根据詹森不等式，对于任意随机变量$`X`$和凸函数$`f`$（下凸，图像像一个坑），有$`f(\mathbb E[X])\le \mathbb E[f(X)]`$；若$`f`$为凹函数（上凸，图像像一个山）则不等式方向反转，有$`f(\mathbb E[X])\ge \mathbb E[f(X)]`$。因为log是凹函数，则有$`\log(\mathbb E[X])\ge \mathbb E[\log(X)]`$。令

```math
Z
\doteq
\frac{
p_{0:T}(x_0,X_{1:T})
}{
q_{1:T}(X_{1:T}\mid x_0)
}
```

则有

```math
\begin{aligned}
\log p_0(x_0)
&=
\log
\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\frac{
p_{0:T}(x_0,X_{1:T})
}{
q_{1:T}(X_{1:T}\mid x_0)
}
\right]
\\
&\geq
\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\log
\frac{
p_{0:T}(x_0,X_{1:T})
}{
q_{1:T}(X_{1:T}\mid x_0)
}
\right]
\end{aligned}
```

我们称大于等于号右侧为证据下界

```math
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x_0)
&=\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\log
\frac{
p_{0:T}(x_0,X_{1:T})
}{
q_{1:T}(X_{1:T}\mid x_0)
}
\right]
\end{aligned}
```

则有
$`\log p_0(x_0)\ge\mathcal{L}_{\mathrm{ELBO}}(x_0)`$
简单介绍一下贝叶斯公式

```math
P(A\mid B)
=
\frac{
P(B\mid A)P(A)
}{
P(B)
}
```

当条件为B,C时有

```math
P(A\mid B,C)
=
\frac{
P(B\mid A,C)P(A\mid C)
}{
P(B\mid C)
},\quad \text{（将 C 作为背景条件）}
```

若将B作为背景条件，则有

```math
P(A\mid B,C)=\frac{
P(C\mid A,B)P(A\mid B)
}{
P(C\mid B)
}
```

因此有

```math
\begin{aligned}
q_{t-1}(X_{t-1}\mid X_t,x_0)
&=
\frac{
q_t(X_t\mid X_{t-1},x_0)q_{t-1}(X_{t-1}\mid x_0)
}{
q_t(X_t\mid x_0)
}
\\&=\frac{
q_t(X_t\mid X_{t-1})q_{t-1}(X_{t-1}\mid x_0)
}{
q_t(X_t\mid x_0)
}\quad \text{（马尔可夫性质）}
\end{aligned}
```

解释一下条件概率链式法则。设$`A,B,C`$为任意随机变量，则其联合概率密度为
$`q(A,B,C)=q(C)q(B\mid C)q(A\mid B,C)`$
本质上来说，即使这些随机变量之间是有依赖顺序的，比如A由B组成，B由C组成，但这不代表上式也只能按这个顺序展开。实际上，我们可以将上式以任何顺序将上式展开，比如
$`q(A,B,C)=q(A)q(B\mid A)q(C\mid A,B)`$
由条件概率的链式法则，我们可以将$`q_{1:T}(X_{1:T}\mid x_0)`$以任意顺序拆解，从而写成另一种形式

```math
\begin{aligned}
q_{1:T}(X_{1:T}\mid x_0)
&=
q_T(X_T\mid x_0)
q_{T-1}(X_{T-1}\mid X_T,x_0)
q_{T-2}(X_{T-2}\mid X_{T-1},X_T,x_0)\cdots
q_1(X_1\mid X_2,X_3,\cdots,X_T,x_0)
\\&=q_T(X_T\mid x_0)
q_{T-1}(X_{T-1}\mid X_T,x_0)
q_{T-2}(X_{T-2}\mid X_{T-1},x_0)
\cdots
q_1(X_1\mid X_2,x_0)\quad \text{（马尔可夫形式）}
\\&=q_T(X_T\mid x_0)
\prod_{t=2}^{T}
q_{t-1}(X_{t-1}\mid X_t,x_0)
\end{aligned}
```

将$`q_{1:T}(X_{1:T}\mid x_0)`$代回$`\mathcal{L}_{\mathrm{ELBO}}`$得到

```math
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x_0)
&= \mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
   \left[
     \log
     \frac{
       p_T(X_T) \prod_{t=2}^T p_{t-1}(X_{t-1}\mid X_t) p_0(x_0\mid X_1)
     }{
       q_T(X_T\mid x_0) \prod_{t=2}^{T} q_{t-1}(X_{t-1}\mid X_t,x_0)
     }
   \right] \\[4pt]
&= \mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
   \bigl[
     \log p_T(X_T)
     + \sum_{t=2}^T \log p_{t-1}(X_{t-1}\mid X_t)
     + \log p_0(x_0\mid X_1)
     \bigr. \\
&\qquad
   \bigl.
     - \log q_T(X_T\mid x_0)
     - \sum_{t=2}^{T} \log q_{t-1}(X_{t-1}\mid X_t,x_0)
   \bigr]
\end{aligned}
```

我们记

```math
\begin{aligned}
A&\doteq\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}\Bigl[\log p_T(X_T) - \log q_T(X_T\mid x_0)\Bigr]
\\
B&\doteq\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\sum_{t=2}^{T}
\left(
\log p_{t-1}(X_{t-1}\mid X_t)
-
\log q_{t-1}(X_{t-1}\mid X_t,x_0)
\right)
\right]
\\C&\doteq\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}\left[\log p_0(x_0\mid X_1)\right]
\end{aligned}
```

则
$`\mathcal{L}_{\mathrm{ELBO}}=A+B+C`$
根据KL散度定义有

```math
D_{\mathrm{KL}}
\left(
Q_T(\cdot\mid x_0)
\|
P_T
\right)
=\mathbb{E}_{X_T\sim Q_T(\cdot\mid x_0)}
\left[
\log \frac{q_T(X_T\mid x_0)}{p_T(X_T)}
\right]
=
\mathbb{E}_{X_T\sim Q_T(\cdot\mid x_0)}
\left[
\log q_T(X_T\mid x_0)
-
\log p_T(X_T)
\right]
```

要求一个随机变量在联合分布中的期望，可以使用边缘分布积分，则有

```math
\begin{aligned}
A&=\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}\Bigl[\log p_T(X_T) - \log q_T(X_T\mid x_0)\Bigr] \\
& = \int q_{1:T}(x_{1:T}\mid x_0)\,\bigl(\log p_T(x_T) - \log q_T(x_T\mid x_0)\bigr)\,dx_{1:T} \\[2mm]
& = \int_{x_T}\!\cdots\!\int_{x_1} q_{1:T}(x_{1:T}\mid x_0)\,\bigl(\log p_T(x_T) - \log q_T(x_T\mid x_0)\bigr)\,dx_1\cdots dx_T \\[2mm]
& = \int_{x_T} \biggl[ \int_{x_{T-1}}\!\cdots\!\int_{x_1} q_{1:T}(x_{1:T}\mid x_0)\,
      \bigl(\log p_T(x_T) - \log q_T(x_T\mid x_0)\bigr)\,dx_1\cdots dx_{T-1} \biggr] dx_T \\[2mm]
& = \int_{x_T} \bigl(\log p_T(x_T) - \log q_T(x_T\mid x_0)\bigr)
      \biggl[ \int_{x_{T-1}}\!\cdots\!\int_{x_1} q_{1:T}(x_{1:T}\mid x_0)\,dx_1\cdots dx_{T-1} \biggr] dx_T \\[2mm]
& = \int q_T(x_T\mid x_0)\,\bigl(\log p_T(x_T) - \log q_T(x_T\mid x_0)\bigr)\,dx_T \\[2mm]
& = \mathbb{E}_{X_T\sim Q_T(\cdot\mid x_0)}\Bigl[\log p_T(X_T) - \log q_T(X_T\mid x_0)\Bigr] \\[2mm]
& = -D_{\mathrm{KL}}\bigl( Q_T(\cdot\mid x_0) \,\|\, P_T \bigr),\quad P_T=\mathcal N(0,1)
\end{aligned}
```

对于B项

```math
\begin{aligned}
B&=\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\sum_{t=2}^{T}
\left(
\log p_{t-1}(X_{t-1}\mid X_t)
-
\log q_{t-1}(X_{t-1}\mid X_t,x_0)
\right)
\right]
\\
&=
\sum_{t=2}^{T}
\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\log p_{t-1}(X_{t-1}\mid X_t)
-
\log q_{t-1}(X_{t-1}\mid X_t,x_0)
\right]
\end{aligned}
```

每项都涉及两个随机变量$`X_{t-1},X_t`$，我们可以先固定$`X_t`$先对$`X_{t-1}`$求期望，然后再对$`X_t`$求期望（原理和A项中调换积分顺序一样）

```math
\begin{aligned}
&\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\log p_{t-1}(X_{t-1}\mid X_t)
-
\log q_{t-1}(X_{t-1}\mid X_t,x_0)
\right]
\\
&=
\mathbb{E}_{X_t\sim Q_t(\cdot\mid x_0)}
\Bigg[
\mathbb{E}_{X_{t-1}\sim Q_{t-1}(\cdot\mid X_t,x_0)}
\left[
\log p_{t-1}(X_{t-1}\mid X_t)
-
\log q_{t-1}(X_{t-1}\mid X_t,x_0)
\right]
\Bigg]
\end{aligned}
```

由KL散度定义有

```math
\begin{aligned}
&D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\\
&=
\mathbb{E}_{X_{t-1}\sim Q_{t-1}(\cdot\mid X_t,x_0)}
\left[
\log q_{t-1}(X_{t-1}\mid X_t,x_0)
-
\log p_{t-1}(X_{t-1}\mid X_t)
\right]
\end{aligned}
```

代回内部期望得到

```math
\begin{aligned}
-
\mathbb{E}_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right]
\end{aligned}
```

则B项为

```math
\begin{aligned}
-
\sum_{t=2}^{T}
\mathbb{E}_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right]
\end{aligned}
```

接着处理最C项，同理，调换积分顺序后，相当于可以只用$`X_1\mid x_0`$的边缘条件分布来求期望

```math
C=\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)}
\left[
\log p_0(x_0\mid X_1)
\right]
=
\mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
\left[
\log p_0(x_0\mid X_1)
\right]
```

将上面得到的结果合并得到ELBO的KL分解

```math
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x_0)
&=
-
D_{\mathrm{KL}}
\left(
Q_T(\cdot\mid x_0)
\|
P_T
\right)
\\
&\quad
-
\sum_{t=2}^{T}
\mathbb{E}_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right]
\\
&\quad
+
\mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
\left[
\log p_0(x_0\mid X_1)
\right]
\end{aligned}
```

由于有
$`\log p_0(x_0)\ge\mathcal{L}_{\mathrm{ELBO}}(x_0)`$
则有
$`-\log p_0(x_0)\le-\mathcal{L}_{\mathrm{ELBO}}(x_0)`$
定义

```math
\begin{aligned}
\mathcal{L}_{\mathrm{VLB}}(x_0)
\doteq
-
\mathcal{L}_{\mathrm{ELBO}}(x_0)&=
D_{\mathrm{KL}}
\left(
Q_T(\cdot\mid x_0)
\|
P_T
\right)
\\
&\quad
+
\sum_{t=2}^{T}
\mathbb{E}_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right]
\\
&\quad
-
\mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
\left[
\log p_0(x_0\mid X_1)
\right]
\end{aligned}
```

因此在训练中，通常通过最小化$`\mathcal{L}_{\mathrm{VLB}}(x_0)`$从而最大化$`\log p_0(x_0)`$的下界。我们记

```math
\begin{aligned}
L_0
&=
-
\mathbb E_{X_1\sim Q_1(\cdot\mid x_0)}
\left[
\log p_0(x_0\mid X_1)
\right]
\\L_{t-1}
&=
\mathbb E_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right],
\qquad
t=2,\dots,T
\\L_T
&=
D_{\mathrm{KL}}
\left(
Q_T(\cdot\mid x_0)
\|
P_T
\right),\quad P_T=\mathcal N(0,1)
\end{aligned}
```

则有

```math
\mathcal L_{\mathrm{VLB}}
=
L_T
+
\sum_{t=2}^{T}L_{t-1}
+
L_0
```

我们先看

```math
L_{t-1}=\mathbb{E}_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right]
```

要计算上式，我们需要解出$`q_{t-1}(X_{t-1}\mid X_t,x_0)`$，由贝叶斯公式得到

```math
\begin{aligned}
q_{t-1}(X_{t-1}\mid X_t,x_0)
=
\frac{
q_t(X_t\mid X_{t-1},x_0)
\,
q_{t-1}(X_{t-1}\mid x_0)
}{
q_t(X_t\mid x_0)
}
&\\=\frac{
q_t(X_t\mid X_{t-1})
\,
q_{t-1}(X_{t-1}\mid x_0)
}{
q_t(X_t\mid x_0)
}
\end{aligned}
```

由于分母不含$`X_{t-1}`$，而$`q_{t-1}(X_{t-1}\mid X_t,x_0)`$是一个关于$`X_{t-1}`$的函数，则有

```math
q_{t-1}(X_{t-1}\mid X_t,x_0)
=C
q_t(X_t\mid X_{t-1})
q_{t-1}(X_{t-1}\mid x_0)
```

由于

```math
\begin{aligned}
q_t(X_t\mid X_{t-1})
&=
\mathcal N
\left(
X_t;
\sqrt{\alpha_t}X_{t-1},
\beta_t
\right)
\\
q_{t-1}(X_{t-1}\mid x_0)
&=
\mathcal N
\left(
X_{t-1};
\sqrt{\bar\alpha_{t-1}}x_0,
1-\bar\alpha_{t-1}
\right)
\end{aligned}
```

高斯分布$`X\sim\mathcal N(\mu,\sigma^2)`$的概率密度函数如下

```math
\mathcal N(x;\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
```

其中$`\exp(A)=e^A`$。条件高斯分布

```math
\mathcal L(Y\mid X=x)
=
\mathcal N
\left(
\mu(x),
\sigma^2(x)
\right)
```

的概率密度函数如下

```math
p(y\mid x)
=
\frac{1}{\sqrt{2\pi\sigma^2(x)}}
\exp
\left(
-
\frac{
\left(
y-\mu(x)
\right)^2
}{
2\sigma^2(x)
}
\right)
```

则有

```math
\begin{gathered}
q_t(X_t \mid X_{t-1})
= \frac{1}{\sqrt{2\pi\beta_t}}
\exp\left(
    -\frac{\bigl(X_t - \sqrt{\alpha_t}\,X_{t-1}\bigr)^2}{2\beta_t}
\right)
\\
q_{t-1}(X_{t-1} \mid x_0)
= \frac{1}{\sqrt{2\pi(1-\bar{\alpha}_{t-1})}}
\exp\left(
    -\frac{\bigl(X_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\,x_0\bigr)^2}{2(1-\bar{\alpha}_{t-1})}
\right)
\end{gathered}
```

因为$`1>\beta_t>0,1>\bar\alpha_t>0`$则有

```math
\begin{aligned}
q_{t-1}(X_{t-1}\mid X_t,x_0)
&=C
\exp
\left(
-\frac{
\left(
X_t-\sqrt{\alpha_t}X_{t-1}
\right)^2
}{
2\beta_t
}
\right)
\\
&\quad\times
\exp
\left(
-\frac{
\left(
X_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0
\right)^2
}{
2(1-\bar\alpha_{t-1})
}
\right)
\\
&=C
\exp
\left(
-\frac{
\left(
X_t-\sqrt{\alpha_t}X_{t-1}
\right)^2
}{
2\beta_t
}
-
\frac{
\left(
X_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0
\right)^2
}{
2(1-\bar\alpha_{t-1})
}
\right)
\end{aligned}
```

展开平方项

```math
\begin{gathered}
\left(
X_t-\sqrt{\alpha_t}X_{t-1}
\right)^2
=
\alpha_t X_{t-1}^2
-
2\sqrt{\alpha_t}X_t X_{t-1}
+
X_t^2
\\
\left(
X_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0
\right)^2
=
X_{t-1}^2
-
2\sqrt{\bar\alpha_{t-1}}x_0 X_{t-1}
+
\bar\alpha_{t-1}x_0^2
\end{gathered}
```

其中只有$`X_{t-1}`$是自变量，则不含$`X_{t-1}`$的项都可以合并为一个常数项，则指数部分为

```math
\begin{aligned}
-\frac{\left(X_t-\sqrt{\alpha_t}X_{t-1}\right)^2}{2\beta_t}
-\frac{\left(X_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\right)^2}{2(1-\bar{\alpha}_{t-1})} &= -\frac{1}{2\beta_t}\bigl(\alpha_t X_{t-1}^2 - 2\sqrt{\alpha_t}X_t X_{t-1} + X_t^2\bigr) \\
&\qquad -\frac{1}{2(1-\bar{\alpha}_{t-1})}\bigl(X_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0 X_{t-1} + \bar{\alpha}_{t-1}x_0^2\bigr) \\[4pt]
&= -\frac{1}{2}\Biggl[
\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)X_{t-1}^2 \\
&\qquad -2\left(\frac{\sqrt{\alpha_t}\,X_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\,x_0}{1-\bar{\alpha}_{t-1}}\right)X_{t-1} + \left(\frac{X_t^2}{\beta_t} + \frac{\bar{\alpha}_{t-1}x_0^2}{1-\bar{\alpha}_{t-1}}\right)\Biggr]
\\&= -\frac{1}{2}\Biggl[
\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)X_{t-1}^2-2\left(\frac{\sqrt{\alpha_t}\,X_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\,x_0}{1-\bar{\alpha}_{t-1}}\right)X_{t-1} + \mathrm{const}\Biggr]
\end{aligned}
```

因为在$`X_0=x_0`$的条件下$`X_{t-1},X_t`$可以建立多元高斯分布（在前面已经建立过联合分布$`X_{1:T}`$了），所以$`X_{t-1}\mid X_t,x_0`$也一定服从高斯分布（在$`X_t=X_t,X_0=x_0`$的条件下$`X_{t-1}`$服从高斯分布），则

```math
\begin{aligned}
q_{t-1}(X_{t-1}\mid X_t,x_0)&=\mathcal N\left(X_{t-1}; \tilde\mu_t(X_t,x_0), \tilde\beta_t \right)
\\&=\frac{1}{\sqrt{2\pi\tilde\beta_t}}\mathrm{exp}\left(-\frac{
\left(
X_{t-1}-\tilde\mu_t(X_t,x_0)
\right)^2
}{
2\tilde\beta_t
}\right)
\end{aligned}
```

这里之所以写$`\tilde\beta_t`$而不写$`\tilde\beta_t(X_t,x_0)`$，是因为后面推导会知道$`\tilde\beta_t`$是一个和条件$`x_0,X_t`$无关的常数。这里省略掉自变量只是为了简洁，不过写作$`\tilde\beta_t(X_t,x_0)`$也不影响推导结果。将指数部分展开得到

```math
-\frac{
\left(
X_{t-1}-\tilde\mu_t(X_t,x_0)
\right)^2
}{
2\tilde\beta_t
}=-\frac12
\left[
\frac{1}{\tilde\beta_t}X_{t-1}^2
-
2\frac{\tilde\mu_t(X_t,x_0)}{\tilde\beta_t}X_{t-1}
\right]
+
\mathrm{const}
```

这里$`\tilde\beta_t`$和前向过程中人为设定的$`\beta`$不同，这里的$`\tilde\beta_t`$表示$`X_{t-1}\mid X_t,x_0`$的方差。对比系数有

```math
\begin{aligned}
\frac{1}{\tilde\beta_t}
&=
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar\alpha_{t-1}}=
\frac{
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
}{
\beta_t(1-\bar\alpha_{t-1})
}
\\&=\frac{\alpha_t-\alpha_t\bar\alpha_{t-1}+1-\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}=\frac{1-\alpha_t\bar\alpha_{t-1}}{\beta_t(1-\bar\alpha_{t-1})},\quad(\beta_t=1-\alpha_t)
\\&=\frac{
1-\bar\alpha_t
}{
\beta_t(1-\bar\alpha_{t-1})
},\quad(\bar\alpha_t=\alpha_t\bar\alpha_{t-1})
\end{aligned}
```

进一步有

```math
\tilde\beta_t
=
\frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t
```

可以看到，$`\tilde\beta_t`$是一个和条件$`x_0,X_t`$无关的常数。再次对比系数有

```math
\begin{aligned}
\frac{\tilde\mu_t(X_t,x_0)}{\tilde\beta_t}
&=
\frac{\sqrt{\alpha_t}}{\beta_t}X_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\\
\tilde\mu_t(X_t,x_0)
&=
\tilde\beta_t
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}X_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\right)
\end{aligned}
```

将$`\bar\beta_t`$代入得到

```math
\begin{aligned}
\tilde\mu_t(X_t,x_0)
&=
\frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}X_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\right)
\\
&=
\frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})
}{
1-\bar\alpha_t
}
X_t
+
\frac{
\sqrt{\bar\alpha_{t-1}}\beta_t
}{
1-\bar\alpha_t
}
x_0
\end{aligned}
```

则有

```math
\begin{aligned}
q_{t-1}(X_{t-1}\mid X_t,x_0)&=\mathcal N\left(X_{t-1}; \frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})
}{
1-\bar\alpha_t
}
X_t
+
\frac{
\sqrt{\bar\alpha_{t-1}}\beta_t
}{
1-\bar\alpha_t
}
x_0, \frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t \right)
\end{aligned}
```

因为

```math
\mathcal L (X_t\mid x_0)
=
\mathcal N
\left(
\sqrt{\bar{\alpha}_t}x_0,
1-\bar{\alpha}_t
\right)
```

即在$`X_0=x_0`$的条件下，有

```math
X_t= \sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\bar\varepsilon_t,\quad \bar\varepsilon_t\sim \mathcal  N(0,1)
```

进一步整理得到

```math
x_0
=
\frac{
X_t-\sqrt{1-\bar\alpha_t}\varepsilon_t
}{
\sqrt{\bar\alpha_t}
}
```

这里可能会让人产生疑问：为什么两个随机变量的运算最终得到了一个常数$`x_0`$？其实这是因为$`X_t`$和$`\varepsilon_t`$并不是两个独立的随机变量，$`X_t`$是由$`x_0`$和$`\varepsilon_t`$构造而成的随机变量，因此在$`X_t`$和$`\varepsilon_t`$的运算过程中，随机的部分被抵消掉了，只剩下了常数。代入$`\tilde\mu_t`$得到

```math
\begin{aligned}
\tilde\mu_t(X_t,x_0)
&=
\frac{
\sqrt{\bar\alpha_{t-1}}\beta_t
}{
1-\bar\alpha_t
}
\cdot
\frac{
X_t-\sqrt{1-\bar\alpha_t}\varepsilon_t
}{
\sqrt{\bar\alpha_t}
}
+
\frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})
}{
1-\bar\alpha_t
}
X_t
\\
&=
\frac{1}{\sqrt{\alpha_t}}
\left(
X_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_t
\right)
\end{aligned}
```

这里即使没有出现$`x_0`$，但由于$`x_0`$是隐含于$`X_t`$中的，因此为了形式统一，依旧写在自变量之中。于是我们就可以得到

```math
\begin{aligned}
q_{t-1}(X_{t-1}\mid X_t,x_0)&=\mathcal N\left(X_{t-1}; \frac{1}{\sqrt{\alpha_t}}
\left(
X_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_t
\right), \frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t \right)
\end{aligned}
```

 在训练时，我们知道$`x_0`$和采样噪声$`\varepsilon_t`$，因此可以计算真实后验$`q_{t-1}(X_{t-1}\mid X_t,x_0)`$；但在生成时，我们只有$`X_t`$（通过$`X_T`$可以确定$`X_{T-1}`$，以此类推最后可以得到$`X_0`$），没有$`x_0`$，也不知道真实噪声$`\varepsilon_t`$。因此需要用神经网络定义的$`p_{t-1}(X_{t-1}\mid X_t)`$去拟合训练时可计算的真实后验$`q_{t-1}(x_{t-1}\mid X_t,x_0)`$。 而这个拟合过程我们唯一所需要知道的就是$`\varepsilon_t`$了，因此只需要让神经网络$`\theta`$学习噪声$`\varepsilon_\theta(X_t,t)`$即可，于是我们可以得到

```math
\begin{aligned}
p_{t-1}(X_{t-1}\mid X_t)&=\mathcal{N}\left( X_{t-1};
\mu_\theta(X_t,t), \frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t \right)
\\\mu_\theta(X_t,t)&=\frac{1}{\sqrt{\alpha_t}}
\left(
X_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\varepsilon_\theta(X_t,t)
\right)
\end{aligned}
```

但在实践中，我们往往不将方差局限

```math
\frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t
```

，而是将其设定为一个可以自由选择的时间常数，因此为了保持更一般的形式，将方差写作$`\sigma^2_t`$

```math
p_{t-1}(X_{t-1}\mid X_t)=\mathcal{N}\left( X_{t-1};
\mu_\theta(X_t,t), \sigma^2_t\right)
```

两个高斯分布的KL散度为

```math
\begin{aligned}
&\quad D_{\mathrm{KL}}\left(\mathcal{N}(\mu_1,\sigma_1^2) \,\|\, \mathcal{N}(\mu_2,\sigma_2^2)\right) \\
&= \int_{-\infty}^{\infty}
\frac{1}{\sqrt{2\pi}\sigma_1} \exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)
\log\frac{
\frac{1}{\sqrt{2\pi}\sigma_1} \exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)
}{
\frac{1}{\sqrt{2\pi}\sigma_2} \exp\!\left(-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right)
} \,dx \\
&= \int_{-\infty}^{\infty}
\frac{1}{\sqrt{2\pi}\sigma_1} \exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)
\left[ \log\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2} \right] dx \\
&= \log\frac{\sigma_2}{\sigma_1} \underbrace{\int p(x)dx}_{=1}
- \frac{1}{2\sigma_1^2} \underbrace{\int (x-\mu_1)^2 p(x)dx}_{=\sigma_1^2}
+ \frac{1}{2\sigma_2^2} \underbrace{\int (x-\mu_2)^2 p(x)dx}_{\mathbb{E}_p[(x-\mu_2)^2]} \\
&= \log\frac{\sigma_2}{\sigma_1} - \frac12
+ \frac{1}{2\sigma_2^2} \mathbb{E}_p\big[(x-\mu_2)^2\big] \\
&= \log\frac{\sigma_2}{\sigma_1} - \frac12
+ \frac{1}{2\sigma_2^2} \Big( \mathbb{E}_p\big[((x-\mu_1)+(\mu_1-\mu_2))^2\big] \Big) \\
&= \log\frac{\sigma_2}{\sigma_1} - \frac12
+ \frac{1}{2\sigma_2^2} \Big( \mathbb{E}_p[(x-\mu_1)^2] + 2(\mu_1-\mu_2)\underbrace{\mathbb{E}_p[x-\mu_1]}_{=0} + (\mu_1-\mu_2)^2 \Big) \\
&= \log\frac{\sigma_2}{\sigma_1} - \frac12
+ \frac{1}{2\sigma_2^2} \big( \sigma_1^2 + (\mu_1-\mu_2)^2 \big) \\
&= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12
\end{aligned}
```

此时我们终于可以写出$`L_{t-1}`$中KL散度的解析式了

```math
\begin{aligned}
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
&=
\log
\frac{\sigma_t}{\sqrt{\tilde\beta_t}}
+
\frac{
\tilde\beta_t
+
\left(
\tilde\mu_t(X_t,x_0)-\mu_\theta(X_t,t)
\right)^2
}{
2\sigma_t^2
}
-\frac12
\end{aligned}
```

其中

```math
\begin{aligned}
\tilde\mu_t(X_t,x_0)-\mu_\theta(X_t,t)
&=
\frac{1}{\sqrt{\alpha_t}}
\left(
-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_t
+
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(X_t,t)
\right)
\\
&=
\frac{\beta_t}{
\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}
}
\left(
\varepsilon_\theta(X_t,t)-\varepsilon_t
\right)
\end{aligned}
```

因此

```math
\left(
\tilde\mu_t(X_t,x_0)-\mu_\theta(X_t,t)
\right)^2
=
\frac{
\beta_t^2
}{
\alpha_t(1-\bar\alpha_t)
}
\left(
\varepsilon_t-\varepsilon_\theta(X_t,t)
\right)^2
```

则

```math
\frac{
\left(
\tilde\mu_t(X_t,x_0)-\mu_\theta(X_t,t)
\right)^2
}{
2\sigma_t^2
}
=
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar\alpha_t)
}
\left(
\varepsilon_t-\varepsilon_\theta(X_t,t)
\right)^2
```

则

```math
\begin{aligned}
L_{t-1}
&=
\mathbb E_{X_t\sim Q_t(\cdot\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
Q_{t-1}(\cdot\mid X_t,x_0)
\|
P_{t-1}(\cdot\mid X_t)
\right)
\right]
\\
&=
\mathbb E_{\varepsilon_t\sim\mathcal N(0,1)}
\left[
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar\alpha_t)
}
\left(
\varepsilon_t-\varepsilon_\theta(X_t,t)
\right)^2
\right]
+
C_t ,\quad C_t
=
\log
\frac{\sigma_t}{\sqrt{\tilde\beta_t}}
+
\frac{\tilde\beta_t}{2\sigma_t^2}
-\frac12
\end{aligned}
```

因为$`X_t`$由$`\varepsilon_t`$和$`x_0`$决定，因此期望下标写明了对$`\varepsilon_t`$求期望，就不需要再写$`X_t\sim Q_t(\cdot\mid x_0)`$了。在优化过程中常数可以省略，于是$`L_{t-1}`$等价于

```math
L_{t-1}'=\mathbb E_{\varepsilon_t\sim\mathcal N(0,1)}
\left[
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar\alpha_t)
}
\left(
\varepsilon_t-\varepsilon_\theta(X_t,t)
\right)^2
\right]
```

将$`X_t= \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon_t`$代入得到

```math
L_{t-1}'=\mathbb E_{\varepsilon_t\sim\mathcal N(0,1)}
\left[
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar\alpha_t)
}
\left(
\varepsilon_t-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon_t,t)
\right)^2
\right]
```

我们继续看$`\mathcal L_{\mathrm{VLB}}`$中的

```math
L_T=D_{\mathrm{KL}}
\left(
Q_T(\cdot\mid x_0)
\|
P_T,\quad P_T=\mathcal L(X_T)
\right)
```

。其中

```math
\begin{aligned}
P_T&=\mathcal N(0,1)
\\Q_T(\cdot\mid x_0)
&=
\mathcal N
\left(
\sqrt{\bar\alpha_T}x_0,
1-\bar\alpha_T
\right)
\end{aligned}
```

由于两个分布都可以直接计算，因此

```math
D_{\mathrm{KL}}
\left(
Q_T(\cdot\mid x_0)
\|
P_T
\right)
```

可以直接计算，和神经网络无关（中间项是因为有无法直接计算的$`p_{t-1}`$所以才要引入神经网络），因此在优化过程可以忽略。继续看$`\mathcal L_{\mathrm{VLB}}`$中的

```math
L_0=-\mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
\left[
\log p_0(x_0\mid X_1)
\right]
```

。其中

```math
\begin{aligned}
p_0(x_0\mid x_1)&=\mathcal{N}\left( x_0;\mu_\theta(x_1, 1), \sigma^2_1 \right)
\\\mu_\theta(X_1,1)&=\frac{1}{\sqrt{\alpha_1}}
\left(
X_1
-
\frac{\beta_1}{\sqrt{1-\bar\alpha_1}}
\varepsilon_\theta(X_1,1)
\right)
\end{aligned}
```

将

```math
X_1
=
\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1
```

代入得到

```math
\begin{aligned}
\mu_\theta(X_1,1)&=\frac{1}{\sqrt{\alpha_1}}
\left(
\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1
-
\frac{\beta_1}{\sqrt{1-\bar\alpha_1}}
\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)
\right)
\
\\&=x_0+\sqrt{\frac{\beta_1}{\alpha_1}}\varepsilon_1-\frac{\beta_1}{\sqrt{\alpha_1-\alpha_1^2}}\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)
\\&=x_0+\sqrt{\frac{\beta_1}{\alpha_1}}\left(\varepsilon_1-\frac{\sqrt{\beta_1}}{\sqrt{1-\alpha_1}}\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)\right)
\\&=x_0+\sqrt{\frac{\beta_1}{\alpha_1}}\left(\varepsilon_1-\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)\right)\quad (\frac{\sqrt{\beta_1}}{\sqrt{1-\alpha_1}}=1)
\end{aligned}
```

则

```math
\begin{aligned}
L_0(x_0)
&= \mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
   \left[ -\log \mathcal{N}\!\left( x_0; \mu_\theta(X_1,1), \sigma_1^2 \right) \right] \\[4pt]
&= \mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
   \left[ -\log \left( \frac{1}{\sqrt{2\pi\sigma_1^2}}
          \exp\!\left( -\frac{(x_0-\mu_\theta(X_1,1))^2}{2\sigma_1^2} \right) \right) \right] \\[4pt]
&= \mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
   \left[ -\log \frac{1}{\sqrt{2\pi\sigma_1^2}}
          - \log \exp\!\left( -\frac{(x_0-\mu_\theta(X_1,1))^2}{2\sigma_1^2} \right) \right] \\[4pt]
&= \mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
   \left[ \frac{1}{2}\log(2\pi\sigma_1^2)
          - \left( -\frac{(x_0-\mu_\theta(X_1,1))^2}{2\sigma_1^2} \right) \right] \\[4pt]
&= \mathbb{E}_{X_1\sim Q_1(\cdot\mid x_0)}
   \left[ \frac{1}{2}\log(2\pi\sigma_1^2)
          + \frac{(x_0-\mu_\theta(X_1,1))^2}{2\sigma_1^2} \right]\\[4pt]
&= \mathbb{E}_{\varepsilon_1\sim \mathcal N(0,1)}
   \left[ \frac{1}{2}\log(2\pi\sigma_1^2)
          + \frac{\left(\sqrt{\frac{\beta_1}{\alpha_1}}\left(\varepsilon_1-\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)\right)\right)^2}{2\sigma_1^2} \right]\\[4pt]
&= \mathbb{E}_{\varepsilon_1\sim \mathcal N(0,1)}
   \left[ \frac{1}{2}\log(2\pi\sigma_1^2)
          + \frac{\beta_1\left(\varepsilon_1-\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)\right)^2}{2\alpha_1\sigma_1^2} \right]
\end{aligned}
```

其中，在机器学习或数学领域， $`\log`$都默认指以$`e`$为底数的$`\log`$函数，所以才有$`\log\exp\!\left( -\frac{(x_0-\mu_\theta(X_1,1))^2}{2\sigma_1^2} \right)=\! -\frac{(x_0-\mu_\theta(X_1,1))^2}{2\sigma_1^2}`$。优化过程中可以忽略常数，则$`L_0`$等价于

```math
\begin{aligned}
L_0'
&=
\mathbb E_{\varepsilon_1\sim\mathcal N(0,1)}
\left[\frac{\beta_1\left(\varepsilon_1-\varepsilon_\theta(\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\varepsilon_1,1)\right)^2}{2\alpha_1\sigma_1^2}
\right]
\\&=\mathbb E_{\varepsilon_1\sim\mathcal N(0,1)}
\left[\frac{\beta_1\left(\varepsilon_1-\varepsilon_\theta(\sqrt{\bar\alpha_1}x_0
+
\sqrt{1-\bar\alpha_1}\varepsilon_1,1)\right)^2}{2\alpha_1\sigma_1^2}
\right],\quad \alpha_t= 1-\beta_t,\, \bar{\alpha}_t = \prod_{i=1}^t \alpha_i
\end{aligned}
```

因此优化完整的$`\mathcal L_{\mathrm{VLB}}`$等价于优化

```math
\begin{aligned}
\mathcal L_{\mathrm{VLB}}&=L_0'+\sum_{t=2}^{T}L_{t-1}'
\\&=\mathbb E_{\varepsilon_1\sim\mathcal N(0,1)}
\left[\frac{\beta_1}{2\alpha_1\sigma_1^2}\left(\varepsilon_1-\varepsilon_\theta(\sqrt{\bar\alpha_1}x_0
+
\sqrt{1-\bar\alpha_1}\varepsilon_1,1)\right)^2\right]
\\&+\sum_{t=2}^T \mathbb E_{\varepsilon_t\sim\mathcal N(0,1)}
\left[
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar\alpha_t)
}
\left(
\varepsilon_t-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon_t,t)
\right)^2
\right]
\end{aligned}
```

但在DDPM的实践中发现，若将$`L_t'`$中的复杂权重去掉，将$`L_0'`$也统一起来，则只优化如下简化后的损失函数，模型的样本生成质量会更高

```math
L_{\mathrm{simple}}
=\sum_{t=1}^T
\mathbb E_{\varepsilon\sim \mathcal N(0,1)}
\left[
\left(
\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\varepsilon,t)
\right)^2
\right]
```

加入一个常数$`\frac{1}{T}`$不影响优化

```math
\begin{aligned}
L_{\mathrm{simple}}
&=\sum_{t=1}^T\frac{1}{T}
\mathbb E_{\varepsilon\sim \mathcal N(0,1)}
\left[
\left(
\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\varepsilon,t)
\right)^2
\right]
\\
&=\mathbb E_{t\sim \text{Uniform}(\{1,\cdots,T\})}\left[\mathbb E_{\varepsilon\sim \mathcal N(0,1)}
\left[
\left(
\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\varepsilon,t)
\right)^2
\right]\right]
\\
&=\mathbb E_{\varepsilon\sim \mathcal N(0,1),t\sim \text{Uniform}(\{1,\cdots,T\})}
\left[
\left(
\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\varepsilon,t)
\right)^2
\right]
\end{aligned}
```

其中$`\text{Uniform}(\{1,\cdots,T\})`$表示均匀分布。 当训练集中只有$`x_0`$时，训练数据分布为$`\mathcal N(x_0,0)`$。我们随机采样$`t、\epsilon`$，计算得到

```math
x_t=\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
```

，将$`x_t,t`$输入神经网络$`\theta`$得到输出$`\epsilon_\theta`$，此时我们就具备了进行随机梯度下降所需的所有数据，重复这个过程，直到训练结束。在生成阶段，输入$`x_t,t`$给神经网络$`\theta`$，得到输出$`\epsilon_\theta`$，接着由

```math
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\varepsilon_\theta(x_t,t)
\right)
```

可以计算得到$`\mu_\theta`$，最后根据

```math
p_{t-1}(x_{t-1}\mid x_t)=\mathcal{N}\left( x_{t-1};
\mu_\theta(x_t,t), \sigma^2_t\right)
```

可以得到生成$`x_{t-1}`$的分布，重复这个过程，最终可以得到生成$`x_0`$的分布。由于$`x_0`$的分布是$`\mathcal N(x_0,0)`$，因此由神经网络给出的生成$`x_0`$的分布也大概会是$`\mathcal N(x_0,0)`$。假如现在有一张64*64*3的图片，将其展平为$`x_0\in \mathbb R^{64*64*3}`$，同时将$`\varepsilon`$也扩展为$`\varepsilon\in \mathbb R^{64*64*3}`$，则损失函数就变为了

```math
L_{\mathrm{simple}}=\mathbb E_{\varepsilon\sim \mathcal N(0,\mathbf I),t\sim \text{Uniform}(\{1,\cdots,T\})}
\left[
\|
\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\varepsilon,t)
\|^2
\right]
```

通过优化该损失函数，神经网络就学会生成图片了。

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics.
