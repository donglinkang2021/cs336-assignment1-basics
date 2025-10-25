$$
\begin{aligned}
\mathrm{SiLU}(x) &= \mathrm{silu}(xW_1)W_2 \\
\mathrm{SiLU1}(x) &= \mathrm{silu}(xW_1)W_1^\top \\
\mathrm{VQ}(x) &= \mathrm{softmax}(xW_1)W_1^\top \\
\mathrm{MHVQ}(x) &= \mathrm{ch}\left(
    \mathrm{softmax}(\mathrm{sh}(x)C_1) C_1^\top
\right) \\
\mathrm{Head}(x) &= \mathrm{ch}(\mathrm{att}(\mathrm{sh}(xW_1))W_2) \\
\mathrm{MHSiLU}(x) &= \mathrm{ch}\left(
    \mathrm{silu}(\mathrm{sh}(x)C_1) C_2
\right) \\
\mathrm{MHSiLU1}(x) &= \mathrm{ch}\left(
    \mathrm{silu}(\mathrm{sh}(x)C_1) C_1^\top
\right) \\
\end{aligned}
$$

其中一些操作的定义如下：

$$
\begin{aligned}
\mathrm{silu}(x) &= x \otimes \mathrm{sigmoid}(x) \\
\mathrm{sh}(x) &= \text{spilt\_heads}(x) \quad \#\text{Shape:} (B,T,D)\to(B,T,h,D/h) \\
\mathrm{att}(x) &= \mathrm{softmax}(xx^T)x \\
\mathrm{ch}(x) &= \text{cat\_heads}(x) \quad \#\text{Shape:} (B,T,h,D/h)\to(B,T,D) \\
\end{aligned}
$$


https://wandb.ai/donglinkang2021-beijing-institute-of-technology/cs336-assignment1-basics/reports/MLP-Experiment--VmlldzoxNDgwODg5MQ?accessToken=wna5t2vs3wqh0ymjzvazwxhpk845hozut43ghj5a81141uw6e14xazlse4ldzud5