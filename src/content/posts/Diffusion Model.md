---
title: "Diffusion Model"
pubDate: "2026-01-15"
modifiedDate: "2026-01-17"
---

对于多种数据模态，我们可以将观察到的数据视为由相关联的潜在变量（用随机变量 z 表示）所表征或生成。表达这一概念的最佳直觉来自柏拉图的洞穴寓言。在这个寓言中，一群人终生被锁在洞穴里，只能看到投射在面前墙上的二维阴影，这些阴影是由火堆前经过的不可见三维物体产生的。对这些人而言，他们观察到的一切实际上都是由他们永远无法看到的高维抽象概念所决定的。
同样地，我们在现实世界中遇到的物体也可能是某种更高层次表征的产物；例如，这些表征可能封装了颜色、大小、形状等抽象属性。那么，我们所观察到的内容就可以被理解为这些抽象概念的三维投射或实例化，就像洞穴人观察到的实际上是三维物体的二维投影一样。尽管洞穴人永远无法看到（甚至完全理解）那些隐藏的物体，他们仍能对其进行推理和推断；类似地，我们也可以近似地描述那些解释观测数据的潜在表征。
柏拉图洞穴寓言将潜变量的概念阐释为决定可观测现象的潜在不可见表征，但需注意的是，在生成式建模中，我们通常寻求学习低维而非高维的潜在表征。这是因为在没有强先验知识的情况下，试图学习比观测维度更高的表征是徒劳无功的。另一方面，学习低维潜变量也可视为一种数据压缩形式，并可能揭示出描述观测数据的语义化结构。
## ELBO
从数学角度，我们可以将潜在变量和观测数据视为由联合分布 p(x,z) 建模。回顾基于似然的生成建模方法，其核心是通过学习模型来最大化所有观测数据 x 的似然值 p(x) 。我们有两种方式可以处理这个联合分布以还原纯观测数据 p(x) 的似然性：既可以显式地边缘化潜在变量 z ：
$$
p(\pmb {x}) = \int p(\pmb {x},z)dz \tag{1}
$$
或者，我们也可以运用概率链式法则：
$$
p(\pmb {x}) = \frac{p(\pmb{x},z)}{p(z|\pmb{x})} \tag{2}
$$
直接计算并最大化似然函数 p(x) 存在困难，因为这要么涉及对公式 1 中所有潜变量 z 进行积分（对于复杂模型而言难以处理），要么需要获得公式 2 中的真实潜变量编码器 p(z∣x) 。然而，利用这两个公式，我们可以推导出一个称为证据下界（ELBO）的项，顾名思义，这是证据的下界。此处的证据量化为观测数据的对数似然（即$\log p(x)$ )。于是，最大化 ELBO 就成为了优化潜变量模型的替代目标；在最理想情况下，当 ELBO 具有强大的参数化能力且被完美优化时，它将完全等同于证据本身。ELBO 的正式表达式为：
$$
\mathbb{E}_{q_{\phi}(z|x)}\left[\log \frac{p(x,z)}{q_{\phi}(z|x)}\right] \tag{3}
$$
注意此处用$q_{\phi}(z|x)$ 代替了概率链式法则中的$p(z|x)$，即使用一个参数化的函数来近似真实的$p(z|x)$ 。通过公式2推导得到ELBO的过程为
$$
\begin{align*}
\log p(\boldsymbol{x}) &= \log p(\boldsymbol{x}) \int q_\phi(\boldsymbol{z}|\boldsymbol{x}) d\boldsymbol{z} && (\text{Multiply by } 1 = \int q_\phi(z|\boldsymbol{x})dz) \quad &(9) \\
&= \int q_\phi(\boldsymbol{z}|\boldsymbol{x})(\log p(\boldsymbol{x}))d\boldsymbol{z} && (\text{Bring evidence into integral}) \quad &(10) \\
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} [\log p(\boldsymbol{x})] && (\text{Definition of Expectation}) \quad &(11) \\
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z}|\boldsymbol{x})} \right] && (\text{Apply Equation 2}) \quad &(12) \\
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})q_\phi(\boldsymbol{z}|\boldsymbol{x})}{p(\boldsymbol{z}|\boldsymbol{x})q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && \left(\text{Multiply by } 1 = \frac{q_\phi(z|\boldsymbol{x})}{q_\phi(z|\boldsymbol{x})}\right) \quad &(13) \\
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] + \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{q_\phi(\boldsymbol{z}|\boldsymbol{x})}{p(\boldsymbol{z}|\boldsymbol{x})} \right] && (\text{Split the Expectation}) \quad &(14) \\
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] + D_{\text{KL}}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}|\boldsymbol{x})) && (\text{Definition of KL Divergence}) \quad &(15) \\
&\ge \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && (\text{KL Divergence always } \ge 0) \quad &(16)
\end{align*}
$$
这里的推导思路为先通过概率积分和为1引入$q_{\phi}(z|x)$ ，再将$p(x)$ 按照公式2分解，通过添加$q_{\phi}(z|x)$项再分别组合得到分开的两项，后面这项恰好是$q_{\phi}(z|x)$和$p(z|x)$之间的KL散度。考虑在数据给定的情况下数据的似然$\log p(x)$可视为一个常数项。公式15的两项和为常数，想让第二项尽可能小，即让参数化的函数和真实的后验分布尽可能相同，就要让第一项尽可能大，因此最大化第一项等同于最大化$\log p(x)$这个似然值。

## VAE
VAE是AE的一个进阶版本，其保留了AE的思想，希望模型可以将数据通过在一个降维的空间内表示出来，在通过解码器将低维变量恢复为数据本身从而达到降维的效果。对于刚才的ELBO，$p(x,z)$仍然是一个很难得到的函数，因此我们希望通过对其进行进一步分解，并引入一些新的假设来实现能够最大化ELBO的目标，可对ELBO进行以下变换
$$
\begin{align*}
\mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right]
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p_\theta(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && (\text{Chain Rule of Probability}) \quad (17) \\
&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} [\log p_\theta(\boldsymbol{x}|\boldsymbol{z})] + \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && (\text{Split the Expectation}) \quad (18) \\
&= \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} [\log p_\theta(\boldsymbol{x}|\boldsymbol{z})]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}))}_{\text{prior matching term}} && (\text{Definition of KL Divergence}) \quad (19)
\end{align*}
$$

$p_\theta(\boldsymbol{x}|\boldsymbol{z})$是参数化的函数用于逼近真实的$p(x|z)$。此处我们可以对$p(z)$引入先验假设，一般假设$p(z)$是标准正态分布，这里引入先验假设的目的是便于计算公式19的先验匹配项，对于$q_\phi(\boldsymbol{z}|\boldsymbol{x})$，一般假设为具有对角协方差矩阵的多元高斯分布，这样在计算时，KL散度项可通过高斯分布的公式直接计算出解析解，重构项则可以通过蒙特卡罗方法近似。此时目标函数可以改写为
$$
\arg \max_{\phi ,\theta}\mathbb{E}_{q_{\phi}(z|x)}\left[\log p_{\theta}(x|z)\right] - D_{\mathrm{KL}}(q_{\phi}(z|x)\parallel p(z))\approx \arg \max_{\phi ,\theta}\sum_{l = 1}^{L}\log p_{\theta}(x|z^{(l)}) - D_{\mathrm{KL}}(q_{\phi}(z|x)\parallel p(z)) \tag{22}
$$
这里$\{z^{(l)}\}_{l = 1}^{L}$ 是从 $q_{\phi}(z|x)$中随机采样得到的 ，但随机采样过程本身是一般是不可微的，因此无法进行后续训练，但由于我们假设$q_{\phi}(z|x)$是多元高斯分布，因此可以通过重参数化的方法解决这一问题。重参数化将随机变量改写为噪声变量的确定性函数，例如，来自均值为$\mu$、方差为 $\sigma^2$的正态分布 $x\sim \mathcal{N}(x;\mu ,\sigma^2)$的样本可表示为：
$$
x = \mu +\sigma \epsilon \quad \mathrm{with} \epsilon \sim \mathcal{N}(\epsilon ;0,\mathrm{I})
$$
因此z的采样过程可以表示为
$$
z = \mu_{\phi}(x) + \sigma_{\phi}(x)\odot \epsilon \quad \mathrm{with} \epsilon \sim \mathcal{N}(\epsilon ;0,\mathbf{I})
$$

## HVAE
层次化变分自编码器（HVAE）是 VAE 的泛化形式，它将潜在变量扩展到多层级结构。在该框架下，潜在变量本身被解释为由其他更高层次、更抽象的潜在变量生成。
尽管在具有 T 个层级的一般 HVAE 中，每个潜变量都可以依赖于之前所有的潜变量，但本工作聚焦于一种特殊情形——我们称之为马尔可夫 HVAE（MHVAE）。在 MHVAE 中，生成过程是一个马尔可夫链；也就是说，层级间的每次转移都具有马尔可夫性，其中
![](https://pic2.imgdd.cc/item/689c6424e65701530c388f6e.png)
每个潜变量 Zt​ 的解码仅依赖于前一个潜变量 Zt+1​ 。直观而言（如图所示），这可以简单理解为将多个 VAE 逐层堆叠；描述该模型的另一个恰当术语是递归式 VAE。从数学角度，我们将马尔可夫 HVAE 的联合分布和后验分布表示为：
$$
\begin{array}{l}{p(\pmb {x},z_{1:T}) = p(z_{T})p_{\pmb{\theta}}(\pmb {x}|z_{1})\prod_{t = 2}^{T}p_{\pmb{\theta}}(z_{t - 1}|z_{t})}\\ {q_{\phi}(z_{1:T}|\pmb {x}) = q_{\phi}(z_{1}|\pmb {x})\prod_{t = 2}^{T}q_{\phi}(z_{t}|z_{t - 1})} \end{array} \tag{24}
$$
这里就是将上面的ELBO中的两项写为符合当前的马尔可夫HVAE性质的形式。ELBO可以重写为（简单来说，就是将公式3中的两项替换一下）
$$
\begin{array}{rlr}\log p(\pmb {x}) = \log \int p(\pmb {x},z_{1:T})dz_{1:T} & \mathrm{(Apply~Equation~1)}\\ = \log \int \frac{p(\pmb{x},z_{1:T})q_{\phi}(z_{1:T}|\pmb{x})}{q_{\phi}(z_{1:T}|\pmb{x})} dz_{1:T} & \mathrm{(Multiply~by~1 = \frac{q_{\phi}(z_{1:T}|\pmb{x})}{q_{\phi}(z_{1:T}|\pmb{x})})}\\ = \log \mathbb{E}_{q_{\phi}(z_{1:T}|\pmb {x})}\left[\frac{p(\pmb{x},z_{1:T})}{q_{\phi}(z_{1:T}|\pmb{x})}\right] & \mathrm{(Definition~of~Expectation)}\\ \geq \mathbb{E}_{q_{\phi}(z_{1:T}|\pmb {x})}\left[\log \frac{p(\pmb{x},z_{1:T})}{q_{\phi}(z_{1:T}|\pmb{x})}\right] & \mathrm{(Apply~Jensen's~Inequality)} \end{array} \tag{25}
$$
再将式24代入25中的新的ELBO，得到在马尔可夫HVAE情况下的ELBO展示式
$$
\mathbb{E}_{q_{\phi}(z_{1:T}|\pmb {x})}\left[\log \frac{p(\pmb{x},z_{1:T})}{q_{\phi}(z_{1:T}|\pmb{x})}\right] = \mathbb{E}_{q_{\phi}(z_{1:T}|\pmb {x})}\left[\log \frac{p(z_{T})p_{\pmb{\theta}}(\pmb{x}|z_{1})\prod_{t = 2}^{T}p_{\pmb{\theta}}(z_{t - 1}|\pmb{z}_{t})}{q_{\phi}(z_{1}|\pmb{x})\prod_{t = 2}^{T}q_{\phi}(z_{t}|\pmb{z}_{t - 1})}\right] \tag{29}
$$
## VDM
在马尔可夫HVAE的基础上加入三个限制条件
1. 潜在维度严格等于数据维度
2. 每个时间步的潜在编码器结构并非学习得到，而是预定义为线性高斯模型。即该模型是以前一时刻输出为中心的高斯分布
3. 潜在编码器的高斯参数随时间变化，确保最终时间步 T 的潜在分布为标准高斯分布
这里由第三个限制条件可以得知，式29中的$p(z_T)$即为最终时间步T的潜在分布，即这个分布为标准高斯分布，此处考虑到限制条件1，为了表达方便，可以将上面的符号重写，将原始数据表示为$x_0$，将后续的潜变量$z_{1:T}$表示为$x_{1:T}$。则之前的联合分布和后验分布可以写为
$$
\begin{align*}
q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) &= \prod_{t=1}^T q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) \tag{30} \\
p(\boldsymbol{x}_{0:T}) &= p(\boldsymbol{x}_T) \prod_{t=1}^T p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \tag{32}
\end{align*}
$$
其中
$$
\begin{align*}
q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) &= \mathcal{N}(\boldsymbol{x}_t; \sqrt{\alpha_t}\boldsymbol{x}_{t-1}, (1-\alpha_t)\boldsymbol{I}) \tag{31} \\
p(\boldsymbol{x}_T) &= \mathcal{N}(\boldsymbol{x}_T; \boldsymbol{0}, \boldsymbol{I}) \tag{33}
\end{align*}
$$
此处的$\alpha_t$是一个自定义的控制t时刻高斯分布的参数，$\alpha_t$可以直接人为指定，下面是两种常用调度方法: 
**线性调度方法**:即让噪声的方差 `β_t` （$\beta_t=1-\alpha_t$)随着时间步 `t` 从一个很小的值线性增加到一个较大的值。
- **设定方式**:
    - 首先确定总的扩散步数 `T` (例如 `T=1000` 或 `T=4000`)。
    - 然后设定 `β_1` (初始方差) 和 `β_T` (最终方差)。
    - `β_t` 的值通过线性插值得到： `β_t = β_1 + (t/T) * (β_T - β_1)`
- **常见取值**:
    - `β_1 = 1e-4` (0.0001)
    - `β_T = 0.02`
得到$\beta_t$自然也就得到了$\alpha_t$。
**余弦调度方法**:不直接定义 `β_t`，而是先定义 `α_t` 的累积乘积 `ᾱ_t` (alpha-bar)。`ᾱ_t` 代表从 `x_0` 一步加噪到 `x_t` 时，`x_0` 前的系数的平方。我们希望 `ᾱ_t` 从1平滑地过渡到接近0，而不是像线性调度那样在后期急剧下降。
- **设定方式**:
    1. 首先定义一个函数 `f(t)`： `f(t) = cos^2( (t/T + s) / (1 + s) * π/2 )` 其中 `s` 是一个很小的偏移量（例如 `s = 0.008`），用来防止 `t=0` 时 `β_t` 变得过小。
    2. 然后用 `f(t)` 来定义 `ᾱ_t`: `ᾱ_t = f(t) / f(0)`
    3. 最后，通过 `ᾱ_t` 反推出每一个时间步的 `β_t`: `β_t = 1 - ᾱ_t / ᾱ_{t-1}` (因为 `α_t = ᾱ_t / ᾱ_{t-1}` 且 `β_t = 1 - α_t`)
- **特点**:
    - **优点**:
        1. `ᾱ_t` 的下降曲线非常平滑，使得噪声的添加在整个过程中更加均匀。
        2. 避免了线性调度在开始和结束阶段的突变问题，训练更稳定。
        3. 生成的样本质量更高，特别是在使用较少采样步数（例如50步或100步）进行推理时，效果远超线性调度。
则在HVAE中的ELBO用上面的式30和32重写并进行一系列变换后得到
$$
\begin{align*}
\log p(\boldsymbol{x}) &= \log \int p(\boldsymbol{x}_{0:T}) d\boldsymbol{x}_{1:T} \tag{34} \\
&= \log \int \frac{p(\boldsymbol{x}_{0:T})q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}d\boldsymbol{x}_{1:T} \tag{35} \\
&= \log \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \frac{p(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \right] \tag{36} \\
&\ge \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \right] \tag{37} \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_{T}) \prod_{t=1}^T p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{\prod_{t=1}^T q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})} \right] \tag{38} \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_T)p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)\prod_{t=2}^T p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_1|\boldsymbol{x}_0)\prod_{t=2}^T q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right] \tag{39} \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_T)p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)\prod_{t=1}^{T-1} p_\theta(\boldsymbol{x}_{t}|\boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\prod_{t=1}^{T-1} q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right] \tag{40} \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_T)p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)}{q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})} \right] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{\prod_{t=1}^{T-1} p_\theta(\boldsymbol{x}_t|\boldsymbol{x}_{t+1})}{\prod_{t=1}^{T-1} q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right] \tag{41} \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} [\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})} \right] + \sum_{t=1}^{T-1} \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_t|\boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right] \tag{42} \\
&= \mathbb{E}_{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} [\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})} \right] + \sum_{t=1}^{T-1} \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_t|\boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right] \tag{43} \\
&= \mathbb{E}_{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} [\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)] + \mathbb{E}_{q(\boldsymbol{x}_{T-1},\boldsymbol{x}_T|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})} \right] + \sum_{t=1}^{T-1} \mathbb{E}_{q(\boldsymbol{x}_{t-1},\boldsymbol{x}_t,\boldsymbol{x}_{t+1}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_t|\boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right] \tag{44} \\
&= \underbrace{\mathbb{E}_{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} [\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)]}_{\text{reconstruction term}} - \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_0)} [D_{\text{KL}}(q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1}) \| p(\boldsymbol{x}_T))]}_{\text{prior matching term}} \tag{45} \\
& \qquad - \sum_{t=1}^{T-1} \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{t-1},\boldsymbol{x}_{t+1}|\boldsymbol{x}_0)} [D_{\text{KL}}(q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) \| p_\theta(\boldsymbol{x}_t|\boldsymbol{x}_{t+1}))]}_{\text{consistency term}}
\end{align*}
$$
其中43-44是因为期望的表达式中仅含有相关的变量，则其余的无关变量作为边缘概率被消去。44-45可以通过KL散度的定义结合贝叶斯公式得到。
此处式45中的第一项是重构项，表达的是在已知第一步隐变量的情况下，重构得到原始数据的可能性的对数。和VAE中的重构项含义相同。第二项同样和VAE中的先验匹配项含义相同，此处$p(x_T)$根据假设恰好为标准高斯分布。第三项则是一致性约束项，致力于使$x_t$处的分布在正向和反向过程中保持一致。具体表现为：在每个中间时间步，从噪声较多图像的去噪步骤应与从较清晰图像的加噪步骤相匹配，这一关系通过 KL 散度在数学上体现。当我们训练$p_\theta({x}_t|{x}_{t+1})$以匹配公式 31 中定义的高斯分布时，该项达到最小化。
通过这个推导得到的公式，第一项与第二项和刚才VAE中的训练方式相同，第三项可以通过蒙特卡罗方法通过随机采样来优化，但注意到第三项的期望中包含两个随机变量$x_{t-1}$和$x_{t+1}$，则对两个随机变量进行采样的方差会高于仅对一个随机变量采样，而我们需要对T个时间步的采样进行加和，因此最终得到的方差在T很大时可能很大，因此我们可以考虑推导出一个仅包含一个随机变量的ELBO表达式来进行优化以避免过大的方差。根据马尔可夫的性质（$x_t$时刻仅和$x_{t-1}$有关，因此可添加一个条件变量$x_0$不会影响分布）和贝叶斯公式，可以得到
$$
q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)q(\boldsymbol{x}_t|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)} \quad (46)
$$
将其代入之前ELBO的推导过程中替换掉之前的$q(x_t|x_{t-1})$。得到如下推导过程
$$
\begin{align}
\log p(\mathbf{x}) &\ge \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] \tag{47} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{\prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] \tag{48} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)\prod_{t=2}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_1|\mathbf{x}_0)\prod_{t=2}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] \tag{49} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)\prod_{t=2}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_1|\mathbf{x}_0)\prod_{t=2}^T q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)}\right] \tag{50} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} + \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)}\right] \tag{51} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} + \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)q(\mathbf{x}_t|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}}\right] \tag{52} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} + \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}\right] \tag{53} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1|\mathbf{x}_0)}{q(\mathbf{x}_T|\mathbf{x}_0)} + \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)}\right] \tag{54} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_T|\mathbf{x}_0)} + \sum_{t=2}^T \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}\right] \tag{55} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right] + \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)}\right] + \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}\right] \tag{56} \\
&= \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right] + \mathbb{E}_{q(\mathbf{x}_T|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)}\right] + \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_t, \mathbf{x}_{t-1}|\mathbf{x}_0)}\left[\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}\right] \tag{57} \\
&= \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)]}_{\text{reconstruction term}} - \underbrace{D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{\text{prior matching term}} - \sum_{t=2}^T \underbrace{\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}[D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))]}_{\text{denoising matching term}} \tag{58}
\end{align}
$$
其中前两项和刚才相同，最后一项现在只有$x_t$一个随机向量，方差问题得到了解决，前两项的训练方法在HVAE中已经说明过了，现在问题是最后一项具体如何训练。解决这个问题就要考虑$D_{\text{KL}}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$这项如何计算。
这项根据贝叶斯公式可得到
$$
q\left(x_{t-1}\mid x_t, x_0\right)=\frac{q\left(x_t\mid x_{t-1},x_0\right),q\left(x_{t-1}\mid x_0\right)}{q\left(x_t\mid x_0\right)}
$$
该公式中右侧第一项相当于$q(x_t|x_{t-1})$在之前已经定义过了为
$$
x_t=\sqrt{\alpha_t},x_{t-1}+\sqrt{1-\alpha_t},\epsilon,\quad \epsilon\sim\mathcal{N}(0,\mathbf{I})
$$
对于$q(x_{t}|x_0)$，将公式右侧的$x_{t-1}$展开，再将$x_{t-2}$展开，以此类推最终得到
$$
\begin{align}
\mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^* \tag{61} \\
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^*\right) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^* \tag{62} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^* + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^* \tag{63} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{(\sqrt{\alpha_t - \alpha_t\alpha_{t-1}})^2 + (\sqrt{1 - \alpha_t})^2} \boldsymbol{\epsilon}_{t-2} \tag{64} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t} \boldsymbol{\epsilon}_{t-2} \tag{65} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2} \tag{66} \\
&= \dots \tag{67} \\
&= \sqrt{\prod_{i=1}^t \alpha_i} \mathbf{x}_0 + \sqrt{1 - \prod_{i=1}^t \alpha_i} \boldsymbol{\epsilon}_0 \tag{68} \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0 \tag{69} \\
&\sim \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \tag{70}
\end{align}
$$
其中63-64满足两个独立的高斯随机变量之和仍是高斯随机变量，均值为二者均值和，方差为二者方差和的性质。这样就得到了任意的$q(x_t|x_0)$。将其代入之前的贝叶斯公式可以计算出$q(x_{t-1}|x_t,x_0)$。核心思路是将右边三项表示成高斯分布后只保留指数部分，将二项式展开，将仅含有$x_t$和$x_0$的项全部提取出来统一表示为一个常数项，对剩余项使用配方法将其配方得到关于$x_{t-1}$的高斯分布的概率密度函数。
$$
\begin{align}
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) &= \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} \tag{71} \\
&= \frac{\mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_{t-1}, (1-\alpha_t)\mathbf{I})\mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})}{\mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})} \tag{72} \\
&\propto \exp\left\{-\left[\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{2(1-\bar{\alpha}_{t-1})} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{2(1-\bar{\alpha}_t)}\right]\right\} \tag{73} \\
&= \exp\left\{-\frac{1}{2}\left[\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{1-\alpha_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t}\right]\right\} \tag{74} \\
&= \exp\left\{-\frac{1}{2}\left[\frac{(-2\sqrt{\alpha_t}\mathbf{x}_t\mathbf{x}_{t-1} + \alpha_t\mathbf{x}_{t-1}^2)}{1-\alpha_t} + \frac{(\mathbf{x}_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{t-1}\mathbf{x}_0)}{1-\bar{\alpha}_{t-1}} + C(\mathbf{x}_t, \mathbf{x}_0)\right]\right\} \tag{75} \\
&\propto \exp\left\{-\frac{1}{2}\left[-\frac{2\sqrt{\alpha_t}\mathbf{x}_t\mathbf{x}_{t-1}}{1-\alpha_t} + \frac{\alpha_t\mathbf{x}_{t-1}^2}{1-\alpha_t} + \frac{\mathbf{x}_{t-1}^2}{1-\bar{\alpha}_{t-1}} - \frac{2\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{t-1}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right]\right\} \tag{76} \\
&= \exp\left\{-\frac{1}{2}\left[\left(\frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)\mathbf{x}_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right)\mathbf{x}_{t-1}\right]\right\} \tag{77} \\
&= \exp\left\{-\frac{1}{2}\left[\frac{\alpha_t(1-\bar{\alpha}_{t-1}) + 1-\alpha_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\mathbf{x}_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right)\mathbf{x}_{t-1}\right]\right\} \tag{78} \\
&= \exp\left\{-\frac{1}{2}\left[\frac{\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1-\alpha_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\mathbf{x}_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right)\mathbf{x}_{t-1}\right]\right\} \tag{79} \\
&= \exp\left\{-\frac{1}{2}\left[\frac{1 - \alpha_t\bar{\alpha}_{t-1}}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\mathbf{x}_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right)\mathbf{x}_{t-1}\right]\right\} \tag{80} \\
&= \exp\left\{-\frac{1}{2}\left(\frac{1 - \bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\right)\left[\mathbf{x}_{t-1}^2 - 2\frac{\left(\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right)}{\frac{1-\bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}}\mathbf{x}_{t-1}\right]\right\} \tag{81} \\
&= \exp\left\{-\frac{1}{2}\left(\frac{1 - \bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}\right)\left[\mathbf{x}_{t-1}^2 - 2\frac{\left(\frac{\sqrt{\alpha_t}\mathbf{x}_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right)(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_{t-1}\right]\right\} \tag{82} \\
&= \exp\left\{-\frac{1}{2}\left(\frac{1}{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}}\right)\left[\mathbf{x}_{t-1}^2 - 2\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\mathbf{x}_0}{1-\bar{\alpha}_t}\mathbf{x}_{t-1}\right]\right\} \tag{83} \\
&\propto \mathcal{N}(\mathbf{x}_{t-1}; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\mathbf{x}_0}{1-\bar{\alpha}_t}}_{\mu_q(\mathbf{x}_t, \mathbf{x}_0)}, \underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}}_{\boldsymbol{\Sigma}_q(t)}) \tag{84}
\end{align}
$$
得到了采样得到$x_{t-1}$的概率密度函数后，该高斯分布的方差项中仅包含$\alpha$相关的项，而$\alpha$之前已经讨论过，可以直接按一定规则预先定义好，因此这部分是一个常数项，我们要找到$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$和这一项的KL散度相近，那我们同样也可以将$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$建模成一个高斯分布，方差和$x_{t-1}$的方差相同，都是无关的常数，但注意$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$和$x_0$无关，因此要将其建模为关于$x_t$和t的分布。由两个高斯分布之间的KL散度的公式可得
$$
\begin{align}
&\arg \min_{\theta} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)) \\
&= \arg \min_{\theta} D_{KL}(\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q(t)) \| \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_q(t))) \tag{87} \\
&= \arg \min_{\theta} \frac{1}{2}\left[ \log \frac{|\boldsymbol{\Sigma}_q(t)|}{|\boldsymbol{\Sigma}_q(t)|} - d + \text{tr}(\boldsymbol{\Sigma}_q(t)^{-1}\boldsymbol{\Sigma}_q(t)) + (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q(t)^{-1}(\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q) \right] \tag{88} \\
&= \arg \min_{\theta} \frac{1}{2}[\log 1 - d + d + (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T\boldsymbol{\Sigma}_q(t)^{-1}(\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)] \tag{89} \\
&= \arg \min_{\theta} \frac{1}{2}[(\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T\boldsymbol{\Sigma}_q(t)^{-1}(\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)] \tag{90} \\
&= \arg \min_{\theta} \frac{1}{2}[(\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T(\sigma_q^2(t)\mathbf{I})^{-1}(\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)] \tag{91} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \|\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q\|_2^2 \tag{92}
\end{align}
$$
其中
$$
\begin{align}
\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\mathbf{x}_0}{1 - \bar{\alpha}_t} \tag{93} \\
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)}{1 - \bar{\alpha}_t} \tag{94}
\end{align}
$$
此处$\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)$可以通过神经网络进行拟合。那么最终的优化目标就变为
$$
\begin{align}
&\arg \min_{\theta} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)) \nonumber \\
&= \arg \min_{\theta} D_{KL}(\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q(t)) \| \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_q(t))) \tag{95} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \left[ \left\lVert \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)}{1 - \bar{\alpha}_t} - \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\mathbf{x}_0}{1 - \bar{\alpha}_t} \right\rVert_2^2 \right] \tag{96} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \left[ \left\lVert \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)}{1 - \bar{\alpha}_t} - \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\mathbf{x}_0}{1 - \bar{\alpha}_t} \right\rVert_2^2 \right] \tag{97} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \left[ \left\lVert \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} (\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t) - \mathbf{x}_0) \right\rVert_2^2 \right] \tag{98} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \frac{\bar{\alpha}_{t-1}(1 - \alpha_t)^2}{(1 - \bar{\alpha}_t)^2} \left[ \lVert\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t) - \mathbf{x}_0\rVert_2^2 \right] \tag{99}
\end{align}
$$
这里优化目标可以直观解释为对任意时间步t的噪声图像$x_t$，神经网络识图根据噪声图像$x_t$和时间步索引t预测原始图像$x_0$。因此，优化变分扩散模型（VDM）可归结为训练神经网络从任意加噪版本的图像中预测原始真实图像。此外，通过最小化所有时间步上的期望值，可以近似实现对我们推导出的证据下界（ELBO）目标（公式 58）中所有噪声水平求和项的最小化。可以通过在时间步上进行随机采样来优化。即
$$
\begin{equation}
\arg \min_{\theta} \mathbb{E}_{t \sim U\{2,T\}} \left[\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)} \left[D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t))\right]\right] \tag{100}
\end{equation}
$$
这样的好处是训练时无需同时计算T个时间步上的所有损失，而是随机取一个时间步就可以训练。
除此以外，对训练目标还有两种等价的参数化方法。
第一种是对$x_0$进行变换，根据式69可得
$$
\begin{equation}
\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}_t}} \tag{115}
\end{equation}
$$
将其代入
$$
\begin{align}
\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\mathbf{x}_0}{1 - \bar{\alpha}_t} \tag{116} \\
&= \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t} \tag{117} \\
&= \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t + (1 - \alpha_t)\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\alpha_t}}}{1 - \bar{\alpha}_t} \tag{118} \\
&= \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\mathbf{x}_t}{1 - \bar{\alpha}_t} + \frac{(1 - \alpha_t)\mathbf{x}_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \tag{119} \\
&= \left(\frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}}{1 - \bar{\alpha}_t} + \frac{1 - \alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}\right)\mathbf{x}_t - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \tag{120} \\
&= \left(\frac{\alpha_t(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} + \frac{1 - \alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}\right)\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \tag{121} \\
&= \frac{\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \tag{122} \\
&= \frac{1 - \bar{\alpha}_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \tag{123} \\
&= \frac{1}{\sqrt{\alpha_t}}\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \tag{124}
\end{align}
$$
则原始的优化目标可以转换为
$$
\begin{align}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \frac{1}{\sqrt{\alpha_t}}\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \tag{125} \\[2ex]
&\arg \min_{\theta} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)) \nonumber \\
&= \arg \min_{\theta} D_{KL}(\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q(t)) \| \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_q(t))) \tag{126} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \left[ \left\lVert \frac{1}{\sqrt{\alpha_t}}\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) - \frac{1}{\sqrt{\alpha_t}}\mathbf{x}_t + \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 \right\rVert_2^2 \right] \tag{127} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \left[ \left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0 - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \right\rVert_2^2 \right] \tag{128} \\
&= \arg \min_{\theta} \frac{1}{2\sigma_q^2(t)} \left[ \left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}} (\boldsymbol{\epsilon}_0 - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)) \right\rVert_2^2 \right] \tag{129} \\
&= \arg \min_{\theta} \frac{(1 - \alpha_t)^2}{2\sigma_q^2(t)(1 - \bar{\alpha}_t)\alpha_t} \left[ \lVert \boldsymbol{\epsilon}_0 - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \rVert_2^2 \right] \tag{130}
\end{align}
$$
此处，$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)$是一个学习预测源噪声 ϵ0∼N(ϵ;0,I) 的神经网络，该噪声从 x0中决定 xt。因此我们证明，通过预测原始图像 x0来学习 VDM 等同于学习预测噪声；实证研究表明，预测噪声能获得更优的性能。


## AdaLN-Zero
AdaLN-Zero作为Diffusion Transformer中对时间t条件化的信息注入方式。
标准的层归一化是在特征维度上计算的，公式如下
$$
\hat{f} = \frac{f - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$
其中$\gamma$和$\beta$均为可学习的参数，均值方差均在特征维度上计算，命名为layer normalization是出于历史原因，最初提出该归一化方法时是在\[batch, feature]形状的数据上进行的，相对于之前使用的batch normalization，该方法在feature上进行归一化，从二维视角来看即为横向的”层“。
一般情况下，层归一化中的两个可学习参数实际上并不适用，直接将数据归一化为0均值，1方差的数据。
AdaLN为Adapative Layer Normalization，自适应的层归一化，自适应表现在可学习参数$\gamma$和$\beta$现在并非固定的学习到的值，而是变化的，这种变化通过对条件$c$经过一个MLP得到$\gamma$和$\beta$得到。这种改变思路和Mamba1中的将时不变改为时变的思路相似，AdaLN的公式为
$$
\text{AdaLN}(f, c) = \gamma(c) \odot \text{LN}(f) + \beta(c)
$$
这里$\odot$表示逐元素乘法，两个可学习参数由神经网络生成。Zero表示一种特殊的初始化，将$\beta(c)$设置为0，$\gamma(c) \approx 1$，此处$\gamma(c) \approx 1$意指$\gamma(c)$向量中的值全部约为1，目的是让初始化状态下的AdaLN和标准的LayerNorm相同，保证训练初期的数据不受条件影响。
这种初始化方法解决了训练稳定性和效率的问题：
零偏置初始化：通过将 $\beta(c)$ 初始化为零，早期的激活保持无偏——条件调制在开始时处于“关闭”状态。
单位尺度初始化：$\gamma(c)$接近于 1 防止任何初始重缩放，进一步确保特征幅度保持良好行为。
随着优化的进行，模型为不同的$c$识别出有用的尺度/偏移响应，逐渐利用完整的条件表达能力，而不干扰初始收敛。
这种方法适合条件信号c的维度比较小的情况，因为c维度大的情况下通过简单网络来生成调制参数可能不稳定。

## JiT 
可参考[周亦帆博客JiT解读](https://zhouyifan.net/2025/12/03/20251124-jit/)，内容讲解的非常好，这里仅挑选我觉得非常重要的内容再做进一步的解释。
JiT核心思想是，让扩散模型的预测目标从速度变成清晰图像，就能成功训练出一个像素空间的高分辨率DiT，无需对DiT做复杂改进，但根据周亦帆博客中的实验表明，其实真正的贡献是当对图像做Patch的Patch Size比较大（大于等于4）时，直接预测清晰图像x才比预测速度v更好。

第一个重要的点是要明确设计扩散模型时，有两个可以分离的设计空间：
1. 让神经网络预测的目标
2. 求最终预测损失的目标
实际使用中，我们有三个可以选择的目标$x,\epsilon,v$，且三者之间可以互相转换，但要注意，转换过程不是纯线性的，因此转换得到的不同损失之间并不等价。还要注意这里的v计算得比较粗糙，实际上如果使用DDIM类型的采样，v应该是球面上的速度。
![](https://pic1.imgdb.cn/item/695f561bf6739df2e0cdfbef.png)
现在问题即为，选择什么样的目标组合可以得到最优的结果，根据文中的实验，如上表所示，预测x，使用v做损失是最优的。在这种情况下，如果使用x-pred和x-loss，一般可以写作类似下面的形式
```python
input x  
  
x_pred = net()  
loss = mse(x_pred, x)
```
如果换成v-loss，则会变成下面的形式
```python
input x, eps, t  
  
z = add_noise(x, eps)  
v = (x - z) / (1 - t).clamp_min()  
  
x_pred = net()  
v_pred = (x_pred - z) / (1 - t).clamp_min()  
  
loss = mse(v_pred, v)
```
此处可以发现，z其实没有作用，因为真实的v和v_pred都减去了z，和x的区别仅在于分母上的1-t。但注意这里不是线性变换，随着t的增大，即越接近清晰的真实图像，1-t越接近于0，对应loss的权重越高，因此使用不同的损失在转换后相当于对于不同的t给同一个loss给予了不同的权重。

第二个重要的点是为什么文中的这种组合仅在大Patch Size时有效，这里的大Patch Size即ViT中将图片进行Patch使用的Patch Size。根据周亦帆的实验，如果不进行Patch，其实JiT这种组合并没有显著的优势，直接使用v_pred效果就很好。
那么为什么在大Patch Size时，使用x_pred要比使用其他pred效果好很多呢。JiT的论文使用流形假设来解释这个问题，这里借用周亦帆博客中的内容：
>流形假设表示，高维空间的数据集里的数据并不是均匀分布在整个空间里，而是在一个低维流形上。我们可以用一个简单的例子来说明：假设我们有一个灰度值构成的数据集，可能的灰度值是 `[0, 255]`。但是，我们存储这些颜色的数据集用的是 3 通道的 RGB 颜色。虽然 RGB 颜色空间有$256^3$种可能，但实际上灰度值只有$256^1$种，且它们分布在一条直线上。这条直线就是高维 3D 空间里的 1D 流形。
>JiT 还提出了另一个命题：符合流形假设的数据更容易被神经网络预测。而在扩散模型的预测目标中，纯噪声$\epsilon$是不符合流形假设的，因为它均匀分布在整个高维空间。因此，由纯噪声算得的$v$也是不符合流形假设的。只有来自真实数据集的清晰图片符合流形假设。
>为了验证该命题，JiT 开展了一个迷你实验：为了构造出符合流形假设的数据集，作者将一个 2D 图形用一个维度为 D 的随机投影矩阵投影到了 D 维。接着，作者训练了三个预测目标不同的扩散模型，观察哪个模型能够成功预测这个投影后的 D 维数据。结果发现，随着 D 增加，只有 x-prediction 维持不错的预测效果，预测$\epsilon$和$v$都不行。

![](https://pic1.imgdb.cn/item/695f60ecf6739df2e0ce00a6.png)
>如果神经网络与流形假设的理论是对的，那么 x-prediction 应该总是更优的，为什么我们在 JiT 中发现只有大 patch size 时更优呢？作者在论文里没有详细讨论这一点，而我通过之前的知识大概想出了原因。这涉及神经网络的更底层的概念：一个 Transformer 到底预测了什么？

>在学习神经网络的时候，我们会先学全连接网络，再学 CNN, RNN, Transformer。一般教程会说，全连接网络更容易过拟合，而其他网络泛化性更好。但仔细思考后，我们可以更具体地指出全连接网络和其他高级网络的区别：全连接网络用一套参数建模了所有输入到所有输出的关系，换句话说，对于每个输出元素，它用到的参数是不同的。而其他高级网络实际上是在**用同一组参数输出一个元素**，只不过输出某元素时，输入还包含其它元素的信息。

>以 CNN 和 Transformer 为例，我们来验证这个概念。CNN 对每个元素都用同样的卷积核，只不过每个卷积核的输入不同；Transformer 的注意力操作是一个无参数的信息融合操作，其他所有投影层、MLP 全是逐元素生效的。

>神经网络其实只负责输出一个数据元素，而现在的扩散模型 loss 或者交叉熵 loss 都是逐元素计算的。所以，看上去神经网络学习的是整个数据集的分布，但它只需要学到整个联合分布的分解 (factorization)，也就是其中某一项数据的规律即可。

>根据这个假设，我们来尝试解释 patch size 对 DiT 的影响。不加 patch size 时，图像的每个数据元素是一个三通道的像素。单个像素的分布可能非常容易学，不管它是清晰图片，还是由纯噪声计算出的速度。这时，是否符合流形假设不影响学习难度，因为数据本身的维度就低。哪种预测方式更好需要用另外的理论解释。

>而增加 patch size，其实是让单个元素的分布更加复杂了。我们可以忽略 patchify 里的线性层，把 patchify 看成是把$p\times p$个像素在特征维度上拼接，把三通道数据变成$3pp$通道的数据（事实上，FLUX.1 的 patchify 就是这么做的）。这个通道数为$3pp$的数据才是真正的「高维数据」，Transformer 要预测的输出通道数是$3pp$。这时，就可以根据前面迷你实验的结论，用流形假设来解释为什么$p$较大时 x-prediction 更好。

这里意味着对于CNN和Transformer这种网络来说，我们实际上是通过引入了一些归纳偏置降低了网络的参数量和网络复杂度，因此网络会试图学习一些更通用的特征，但同时这也使得网络试图学习的更通用特征不能在高维空间中过于发散，因为如果分布在高维空间中过于发散，这种类型的网络很难学会这个高维分布，而如果这些“通用特征”满足流形假设，要学习的特征其实分布在一个低维流形上，那么网络就可以学的更好，从而表现为学到了“通用特征”。
这里讨论了大Patch可能带来的高维分布难以学习的问题，但我们之前在时序和图像里面广泛使用Patch是因为Patch本身除了能降低计算量之外还有其他优势，如Patch可以更好的表示局部相似性，可以一定程度上消除局部噪声，因此Patch是一个很有效的降低计算量的手段，但Patch大小的选择需要更好的考量，找到一个效率和效果之间的平衡点。

## Stable Diffusion 3 
主要论述了流匹配使用rectified flow替代一般的DDPM的可行性，通过推导证明了流匹配可以转换成一般的DDPM但是流匹配更高效。此处专指**1-Rectified Flow**，它的流程就是：
1. 随机抽一个噪声 $x_0$。
2. 随机抽一张图 $x_1$。
3. 强行认为它俩之间是一条直线：$z_t = t x_1 + (1-t) x_0$。
4. 训练模型 $v_\theta(z_t, t)$ 去预测方向 $x_1 - x_0$。
这里可能会有疑问，即抽取的$(x_0,x_1)$对之间可能在空间中存在路径交叉，但实际上在现有模型的训练过程中都是采用这样的过程效果也很好，虽然在原文中没有对这一问题做进一步的阐释，但我个人理解可能是因为在图片上训练时空间维度较高，因此随机抽样得到的从噪声到真实图像之间的路径在高维空间下并不容易交叉。相当于高维空间下有足够的空间体积，可以容纳下相当数量的路径集合而不至于存在相交路径，换个角度说路径相交的概率极小，因此实际得到的模型效果都还不错。
另一个则是SD3提出了一种新的信噪比采样器，以前在训练过程中选取采样的时间t是在0-1之间均匀采样的，作者认为在$t=0$和$t=1$时，训练更容易，因为当t=0时，噪声占据主导，此时模型只需要预测一个大概方向，而t=1时图像已经基本接近真实图像，因此中间部分学习起来比较困难。应该更频繁的采样中间时间步让模型更多次数的学习中间部分的样本。这里的Logit-Normal采样法为首先在正态分布上采样，再通过sigmoid函数变换到(0,1)区间得到t的采样分布（Logit-Normal含义为将最终得到分布进行Logit即得到一个正态分布），这样得到的分布t=0.5附近的采样次数比较多。除此以外还要给最终loss一个额外的权重$\frac{t}{1-t}$，这个权重的作用是使得t=1附近的权重增大，以提高最终生成的图像的细节，同时这个权重还能使得训练过程中梯度的方差保持恒定，作者证明了对于直线路径的Flow matching，当权重约为$\frac{1}{1-t}$（信噪比倒数）时，训练过程中方差最小。

## RAE 
VAE组件比较老旧，社区考虑使用一些更好的组件来替换VAE。
RAE论文中提到VAE存在的问题有三点：
- **过时的骨干网络使架构比实际需要的更复杂**：SD-VAE 的计算量约为 450 GFLOPs，而一个简单的 ViT-B 编码器只需要大约 22 GFLOPs。
- **过度压缩的潜空间（只有 4 个通道）限制了可存储的信息量**：人们常说压缩带来智能，但这里并非如此：VAE 式压缩实际上作用有限，几乎和原始的三通道像素一样受限。
- **表征能力弱**：由于仅使用重建任务进行训练，VAE 学到的特征很弱（线性探针精度约 8%，线性探针为冻结模型编码器部分后，将输入一张图片得到的中间特征向量只加一层简单的线性分类层，再在一个数据集上训练这个模型看它的分类准确率，以此表征编码器提取图片语义信息的能力，也就意味着VAE实际上更像是纯粹的压缩解压缩工具），这会导致模型收敛更慢、生成质量下降。我们现在已经很清楚 —— 表征质量直接影响生成质量，而 SD-VAE 并不是为此而设计的。
因此使用预训练好的表征编码器（DINO,SigLIP,MAE）作为特征提取模块，再设计一个解码器，通过训练这个编码器解码器的组合，可以替代传统的VAE，形成文中提到的表征自编码器，相比于VAE这种模型既能实现高质量的重建，也能提供语义丰富的潜空间，同时便于Transformer扩展。
问题在于预训练的表征编码器编码得到的特征是高维特征，而VAE编码器编码得到的是低维潜空间，在高维特征上训练一个DiT存在计算效率，架构，噪声调度等问题，如经典的同样的噪声调度对于高分辨率图像（高维）的噪声强度和低维的效果不同。

## DDT [@wangDDTDecoupledDiffusion2025]
这篇文章提出了解耦的DiT，核心在于目前的扩散模型中使用的主要为Transformer的仅解码器结构，作者想要探究解耦的编码器-解码器结构能够加速收敛，提升样本质量。
![](https://pic1.imgdb.cn/item/696739ca99f37a647f57810f.png)
本文的灵感在于，通过实验发现，在推理阶段给**高噪阶段**（即低频语义占主导的阶段）分配更多计算量效果更好。这证明了当前的 Diffusion Transformer（如 DiT）的短板不在于画不出细节，而是在于**对低频语义结构的编码能力不足**。如图中所示，在推理阶段在较小的t部分分配更多的推理步骤可以降低推理得到的FID，获得更好的图像生成效果。这一理论和之前的REPA使用与预训练视觉基础模型（DINO）的表示对其方法增强低频编码从而取得更好的图像生成效果相符，REPA的表示对齐技术可以表示为
$$
\mathcal{L}_{enc} = 1 - \cos(r_*, h_\phi(\mathbf{h_i}))
$$
该方法将自映射编码器中第 $i$ 层的中间特征 $\mathbf{h}_i$ 与 DINOv2 表示 $r_*$ 对齐。$h_\phi$ 是可学习的投影 MLP。REPA可以加快模型的训练速度，相当于引导模型学会从噪声图像中生成含有语义信息的特征块，类似知识蒸馏，很像教师-学生模型。也正因如此，相近的时间步编码器产生的语义信息应该是相似的（均和REPA对当前图像的语义编码相似），可以共享，因此DDT的编码器部分产生的自条件$z_t$应该在相邻步骤中是可以共享的，减少了计算量，提升了推理效率。
![](https://pic1.imgdb.cn/item/696b1fbc55fa3078186eaedc.png)
通过解耦的条件编码器，速度解码器组件，并且在条件编码器中使用REPA技术，使得条件编码器可以很好的编码图像的语义特征，随后将编码得到的自条件替换原本应该输入给去噪模型的标签，其他输入和之前相同输入给速度解码器，解码器去噪得到速度$v_t$。相当于通过显式的条件编码，将语义信息更好的表示出来，而且实际实现时条件编码器的层数要远多于速度解码器，给条件编码器更大的参数量这一设计思路符合之前实验证实的DiT对语义信息的编码能力不足这一现象，通过给语义编码提供更大的参数量来获得更好的语义信息。
条件编码器得到的语义特征可以在相邻几次的解码过程中重复使用，相当于先构思，再画图，而构思是可以复用一小段时间的，因此最终推理的速度更快，根据文章的仓库的代码实现，语义特征复用也仅在采样和推理阶段开启，训练阶段不使用。

## Mamba
