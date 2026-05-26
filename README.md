加噪过程
假设我们有一张彩色图，其中某个像素的某个颜色通道的值为$x_0$。则加噪过程如下

```math
x_1
=
\sqrt{1-\beta_1}x_0
+
\sqrt{\beta_1}\epsilon_1
```

其中$\epsilon_1$是噪声值，采样于标准高斯变量$\varepsilon\sim\mathcal N(0,1)$，$\beta_1$是人为设定的第1步加噪的噪声强度，一般要求$0<\beta<1$。写成随机变量形式

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

严格来讲$X_t$服从条件高斯分布

```math
\mathcal L\left(X_t\mid X_{t-1}=x_{t-1},X_0=x_0\right)
=\mathcal N
\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_t
\right)
```

由于$X_t$仅依赖上个时刻的结果，故加噪过程本质是一个马尔可夫过程，则可以将$X_0=x_0$的条件省略

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

我们用$q_t$表示条件概率密度值，则有

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
\alpha_t\doteq 1-\beta_t,\quad 1>\alpha_t>0
\\\bar{\alpha}_t \doteq \prod_{i=1}^t \alpha_i,\quad 1>\bar\alpha_t>0
```

假如我们不指定$x_{t-1}$，则递推式可以写成

```math
X_t
=
\sqrt{\alpha_t}X_{t-1}
+
\sqrt{1-\alpha_t}\varepsilon_t
```

则在给定$X_0=x_0$的条件下，有

```math
\begin{align*}
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
\right),\quad(约定空乘积\sqrt{\prod_{j=t+1}^{t}\alpha_j}=1)
\end{align*}
```

可以看出，在给定$X_0=x_0$的条件下，随机变量$X_t$等价于由多个互相独立的高斯随机变量$\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_t$表示的线性组合。因为独立高斯随机变量的线性组合仍为高斯变量，则$X_t\mid x_0$也是高斯变量。计算其均值和方差

```math
\begin{aligned}
\mathbb E(X_t\mid x_0)
&=
\sqrt{\bar{\alpha}_t}x_0
+
\sum_{i=1}^{t}
\sqrt{\prod_{j=i+1}^{t} \alpha_j}
\sqrt{1 - \alpha_i}
\mathbb E(\varepsilon_i)
\\&=\sqrt{\bar{\alpha}_t}x_0
\end{aligned}
```

```math
\begin{aligned}
\operatorname{Var}
\left(
X_t\mid x_0
\right)
&=
\sum_{i=1}^{t-1}
\operatorname{Var}
\left(
\sqrt{\prod_{j=i+1}^{t}\alpha_j}
\sqrt{1-\alpha_i}
\varepsilon_i
\right)
+
\operatorname{Var}
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

因此我们可以直接写出从$x_0$直接去采样$X_t$的条件分布

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

其中$\bar\varepsilon_t$用于和递推式中的局部噪声$\varepsilon_t$做区分，两个随机变量虽然都服从标准正态分布，但两者是不同的随机变量，两者的关系为

```math
\sqrt{1-\bar\alpha_t}\bar\varepsilon_t=\sum_{i=1}^{t}
\left(
\sqrt{\prod_{j=i+1}^{t} \alpha_j}
\sqrt{1 - \alpha_i}
\varepsilon_i
\right),\quad(约定空乘积\sqrt{\prod_{j=t+1}^{t}\alpha_j}=1)
```

条件概率密度函数为
$q_t(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar \alpha_t}x_{0},(1-\bar \alpha_t))$

直接采样等价于一步步采样的本质含义不是说直接采样得到$x_t$和一步步采样得到$x_t$在值上相同，而是说在经过无穷次采样后，两种方式采样得到$x_t$的统计分布是相同的。当$T\to\infty$时，$\bar\alpha_T\to 0$，则有$q_T(x_T\mid x_0)\approx\mathcal{N}(x_T;0,1)$。

## 生成过程

当给定$x_0$后，后续所有时刻结果的随机变量$X_1,X_2,\cdots,X_T$的解析式都可以直接写出。如果我们可以构造一个神经网络$\theta$去学习根据$x_t$和$t$输出噪声，进而逆向恢复原始图像，则可以实现图像生成。

两个随机变量$X,Y$相互独立时，联合概率密度函数为
$f_{X,Y}(x,y)=f_X(x)\cdot f_Y(y)$

当两随机变量不互相独立时，联合概率密度函数为
$f_{X,Y}(x,y)=f_X(x)\cdot f_{Y\mid X}(y\mid x)=f_Y(y)\cdot f_{X\mid Y}(x\mid y)$

在给定$x_0$后，将后续结果对应的随机变量联合起来

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

也可以称$X_{1:T}$为随机向量。$X,Y$可以组成联合高斯（多元高斯分布）的充分必要条件是：对所有非零线性组合$aX+bY$，结果必须是高斯变量或退化高斯变量。联合高斯随机向量$(X,Y)^T$的概率密度函数为

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
\operatorname{Var}(X) & \operatorname{Cov}(X,Y)\\
\operatorname{Cov}(Y,X) & \operatorname{Var}(Y)
\end{pmatrix}
```

其中$\mu$代表均值向量，$\Sigma$代表$X,Y$的协方差矩阵。

假设某个生成过程正确一步步生成得到了$x_0$，采样值依次为$x_{T},\cdots,x_{1},x_0$。我们记生成过程在时刻$t-1$给定$x_t$的条件下的条件概率分布为$p_{t-1}(x_{t-1}\mid x_t)$，则得到该采样序列的联合概率密度函数为

```math
\begin{aligned}
p_{0:T} (x_{0:T})&=p_T(x_T) p_{T-1}(x_{T-1} \mid x_T) p_{T-2}(x_{T-2} \mid x_{T}, x_{T-1}) \cdots p_{0}(x_0 \mid x_T, \dots, x_{1})
\\&=p_T(x_T) p_{T-1}(x_{T-1} \mid x_T) p_{T-2}(x_{T-2} \mid  x_{T-1}) \cdots p_{0}(x_0 \mid x_{1})\qquad \text{(由马尔可夫性质)}
\\&= p_T(x_T) \prod_{t=1}^T p_{t-1}(x_{t-1} \mid x_t)
\end{aligned}
```

由于$X_T$的先验分布在DDPM标准中固定为标准高斯，即
$p_T(x_T)=\mathcal{N}(x_T;0,1)$

要注意，上面的路径$x_T,x_{T-1},\cdots,x_{0}$只是从$x_T$到$x_0$的其中一条路径，除此之外还包含无数条通往$x_0$的路径，所以最终生成$x_0$的概率密度（边际分布）是对所有可能路径进行积分

$p_0(x_0)=\int p_{0:T}(x_{1:T},x_0) \, dx_{1:T}$

最直接的训练目标，就是最大化$p_0(x_0)$，为了简化运算，一般会转化为最大化$\log p_0(x_0)$。但直接优化$\log p_0(x_0)$是很困难的，因此我们引入已知的前向过程分布$q$来建立下界

```math
\begin{aligned}
p_0(x_0) &= \int p_{0:T}(x_{0:T}) \, dx_{1:T} 
\\&= \int q_{1:T}(x_{1:T} \mid x_0) \cdot \frac{p_{0:T}(x_{1:T},x_0)}{q_{1:T}(x_{1:T} \mid x_0)} \, dx_{1:T}
\\&=\mathbb{E}_{X_{1:T}\sim Q_{1:T}(\cdot\mid x_0)} \left[ \frac{p_{0:T}(x_0,X_{1:T})}{q_{1:T}(X_{1:T} \mid x_0)} \right],\quad p_{0:T}(x_0,X_{1:T})=p_T(X_T) \prod_{t=2}^T p_{t-1}(X_{t-1} \mid X_t)p_0(x_0\mid X_1)
\end{aligned}
```

此时$X_1,X_2,\cdots,X_T$表示服从前向条件分布的联合随机变量。设$X$为随机变量，$f(x)$为其概率密度函数；若将$Y\doteq X_1+X_2$代入概率密度函数$f(x)$就得到$f(Y)$，这实际上是对所有满足$X_1+X_2=Y$的$X_1,X_2$的联合密度求积分，即$f(Y)=\int f(x_1,y-x_1) dx_1$。

令

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

最后一步不等式是根据詹森不等式（Jensen's inequality）得出的，因为$\log$是凹函数。我们称大于等于号右侧为证据下界（Evidence Lower BOund, ELBO）

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
$\log p_0(x_0)\ge\mathcal{L}_{\mathrm{ELBO}}(x_0)$

## 贝叶斯公式与条件概率

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

当条件为$B,C$时有

```math
P(A\mid B,C)
=
\frac{
P(B\mid A,C)P(A\mid C)
}{
P(B\mid C)
},\quad(将C作为背景条件)
```

若将$B$作为背景条件，则有

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
}\quad(马尔可夫性质)
\end{aligned}
```

解释一下条件概率链式法则。设$A,B,C$为任意随机变量，则其联合概率密度为
$q(A,B,C)=q(C)q(B\mid C)q(A\mid B,C)$

本质上来说，即使这些随机变量之间是有依赖顺序的，比如$A$由$B$组成，$B$由$C$组成，但这不代表上式也只能按这个顺序展开。实际上，我们可以将上式以任意顺序拆解，例如
$q(A,B,C)=q(A)q(B\mid A)q(C\mid A,B)$

由条件概率的链式法则，我们可以将$q_{1:T}(X_{1:T}\mid x_0)$以任意顺序拆解，从而写成另一种形式

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
q_1(X_1\mid X_2,x_0)\quad (马尔可夫形式)
\\&=q_T(X_T\mid x_0)
\prod_{t=2}^{T}
q_{t-1}(X_{t-1}\mid X_t,x_0)
\end{aligned}
```

## ELBO的KL分解

将$q_{1:T}(X_{1:T}\mid x_0)$代回$\mathcal{L}_{\mathrm{ELBO}}$得到

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
     + \log p_0(x_0\mid X_1) \\
&\qquad
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
$\mathcal{L}_{\mathrm{ELBO}}=A+B+C$

### A项的推导

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

### B项的推导

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

每项都涉及两个随机变量$X_{t-1},X_t$，我们可以先固定$X_t$先对$X_{t-1}$求期望，然后再对$X_t$求期望（原理和A项中调换积分顺序一样）

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

### C项的推导

接着处理C项，同理，调换积分顺序后，相当于可以只用$X_1\mid x_0$的边缘条件分布来求期望

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

## ELBO的KL分解结果

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
$\log p_0(x_0)\ge\mathcal{L}_{\mathrm{ELBO}}(x_0)$

则有
$-\log p_0(x_0)\le-\mathcal{L}_{\mathrm{ELBO}}(x_0)$

定义变分下界（Variational Lower BOund, VLB）

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

则有
$-\log p_0(x_0)\le\mathcal{L}_{\mathrm{VLB}}(x_0)$

因此在训练中，通常通过最小化$\mathcal{L}_{\mathrm{VLB}}(x_0)$从而最大化$\log p_0(x_0)$的下界。我们记

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

## 后验分布的推导

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

要计算上式，我们需要解出$q_{t-1}(X_{t-1}\mid X_t,x_0)$，由贝叶斯公式得到

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

由于分母不含$X_{t-1}$，而$q_{t-1}(X_{t-1}\mid X_t,x_0)$是一个关于$X_{t-1}$的函数，则有

```math
q_{t-1}(X_{t-1}\mid X_t,x_0)
=C
q_t(X_t\mid X_{t-1})
q_{t-1}(X_{t-1}\mid x_0)
```

其中$C$是归一化常数。由于

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

高斯分布$X\sim\mathcal N(\mu,\sigma^2)$的概率密度函数如下

```math
\mathcal N(x;\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
```

其中$\exp(A)=e^A$。条件高斯分布

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
```

因为$1>\beta_t>0,1>\bar\alpha_t>0$则有

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
```

其中只有$X_{t-1}$是自变量，则不含$X_{t-1}$的项都可以合并为一个常数项，则指数部分关于$X_{t-1}$的二次型系数为

```math
-\frac{\alpha_t}{2\beta_t} - \frac{1}{2(1-\bar\alpha_{t-1})}
= -\frac{1}{2}\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}\right)
= -\frac{1}{2}\left(\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}\right)
```

注意到$\alpha_t = 1 - \beta_t$，所以

```math
\alpha_t(1-\bar\alpha_{t-1})+\beta_t = (1-\beta_t)(1-\bar\alpha_{t-1})+\beta_t = 1-\bar\alpha_{t-1}-\beta_t(1-\bar\alpha_{t-1})+\beta_t
\\= 1 - \bar\alpha_{t-1} + \beta_t\bar\alpha_{t-1} = 1 - \bar\alpha_{t-1}(1-\beta_t) = 1 - \bar\alpha_{t-1}\alpha_t = 1 - \bar\alpha_t
```

因此二次项系数为$-\frac{1-\bar\alpha_t}{2\beta_t(1-\bar\alpha_{t-1})}$，进而可得后验分布的方差为

```math
\sigma_q^2(t) = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
```

一次项系数（关于$X_{t-1}$）为

```math
\frac{\sqrt{\alpha_t}X_t}{\beta_t} + \frac{\sqrt{\bar\alpha_{t-1}}x_0}{1-\bar\alpha_{t-1}}
= \frac{\sqrt{\alpha_t}X_t(1-\bar\alpha_{t-1}) + \sqrt{\bar\alpha_{t-1}}x_0\beta_t}{\beta_t(1-\bar\alpha_{t-1})}
```

根据高斯分布的标准形式，均值为

```math
\mu_q(X_t,x_0) = \frac{1}{2}\sigma_q^2(t) \cdot 2 \cdot \frac{\sqrt{\alpha_t}X_t(1-\bar\alpha_{t-1}) + \sqrt{\bar\alpha_{t-1}}x_0\beta_t}{\beta_t(1-\bar\alpha_{t-1})}
\\= \sigma_q^2(t) \cdot \frac{\sqrt{\alpha_t}X_t(1-\bar\alpha_{t-1}) + \sqrt{\bar\alpha_{t-1}}x_0\beta_t}{\beta_t(1-\bar\alpha_{t-1})}
\\= \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \cdot \frac{\sqrt{\alpha_t}X_t(1-\bar\alpha_{t-1}) + \sqrt{\bar\alpha_{t-1}}x_0\beta_t}{\beta_t(1-\bar\alpha_{t-1})}
\\= \frac{\sqrt{\alpha_t}X_t(1-\bar\alpha_{t-1}) + \sqrt{\bar\alpha_{t-1}}x_0\beta_t}{1-\bar\alpha_t}
\\= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})X_t + \sqrt{\bar\alpha_{t-1}}\beta_t x_0}{1-\bar\alpha_t}
```

因此后验分布为

```math
q_{t-1}(X_{t-1}\mid X_t,x_0) = \mathcal N\left(X_{t-1}; \mu_q(X_t,x_0), \sigma_q^2(t)\right)
```

其中

```math
\mu_q(X_t,x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})X_t + \sqrt{\bar\alpha_{t-1}}\beta_t x_0}{1-\bar\alpha_t}
```

```math
\sigma_q^2(t) = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
```

这是前向过程在给定$X_t$和$x_0$时的后验分布，是一个高斯分布，其参数完全由前向过程的参数（$\alpha_t, \beta_t, \bar\alpha_t$等）确定。
