# DLCV HW1

## Problem 1 Bayes Decision Rule

>$$P(ω_1) = \frac{2}{9}, \quad P(x|ω_1)= \begin{cases} \frac{1}{5} & \text {x $\in$ [0, 5] } \\ 0 & \text{otherwise} \end{cases} $$  
>$$P(ω_2) = \frac{7}{9}, \quad P(x|ω_2)= \begin{cases} \frac{1}{7} & \text {x $\in$ [2, 9] } \\ 0   & \text{otherwise} \end{cases}$$
>assume that the decision boundary at T :
>$$\begin{aligned} &\int_T^\infty P(x|ω_1)P(ω_1)dx + \int_{-\infty}^T P(x|ω_2)P(ω_2)dx \\
&= \frac{2}{9} \cdot \frac{1}{5} \cdot (5-T) + \frac{7}{9} \cdot \frac{1}{7} \cdot (T-2) \\
&= \frac{1}{15}T \end{aligned} $$
>$$(5-T) \geq 0 \quad and \quad (T-2) \geq 0 \quad \to \quad 2 \leq T \leq 5$$
>to minimize the error : $T = 2$ and $P_e = \frac{2}{15}\\$ 
>if $x \geq 2$ we choose $ω_2$; else, we choose $ω_1$

## Problem 2 Principal Component Analysis