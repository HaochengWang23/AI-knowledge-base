# RL basics

强化学习主流可以分为两类：

- **Value-based（值函数）**：（DQN系列（离散动作））：不直接学策略，而是学“这个动作有多好”（Q值）
    
    **Q(s,a)** 表示在状态 s 下采取动作 a 的期望回报
    
    决策方式：a=argamaxQ(s,a)
    
    - 不直接输出动作，通过 Q 值“选最优动作”
    - 不适用于连续工作（无法argmax）
- **Policy-based（策略梯度）:** 直接学习策略 π(a|s), 目标函数：J(θ)=Eπθ[R]
    
    更新方式：
    
    $$
    ∇_θJ=E[∇_θlogπ_θ(a∣s)⋅R]
    $$
    
    直接输出动作分布，可以处理连续动作，探索能力强。
    
    问题：方差大，收敛慢
    
- **Actor-Critic（最主流）**：DDPG / TD3 / SAC / PPO
    
    核心思想：Actor 学策略π(a|s)，Critic 学价值函数辅助优化Q(s,a)或V(s)
    
    Actor 不再用原始回报，而是用 Critic：
    
    $$
    ∇_θJ=E[∇_θlogπ(a∣s)⋅Q(s,a)]
    $$
    
    降低方差，更稳定 （公式含义：如果某个动作带来了高回报 R，就增加它的概率（log π 上升），如果回报低，就减少它的概率）
    

Actor-Critic 是对“策略分布的期望”做优化（连续优化），Value-based 是对“动作空间取最大值”做选择（离散决策**）**

Policy Gradient 使用 Monte Carlo return 来更新策略，虽然无偏但方差很大。而 Actor-Critic 引入 value function 来近似未来回报，用 Q 或 advantage 替代 R，从而显著降低方差，提高训练稳定性，但引入了一定偏差。

V 表示一个状态的整体价值，而 Q 则评估在该状态下某个具体动作的价值。Q 比 V 更细粒度，而 V 可以看作是在策略下对 Q 的期望。

on policy: 用当前策略 π 采样数据，并用这些数据更新 π, 样本利用率低但稳定；

off policy: 可以用历史数据，用一个行为策略 μ 收集数据，用另一个策略 π 学习, 提高效率，但需要处理分布偏移问题。

DDPG / TD3 / SAC 主要用于**连续动作空间，**PPO 更通用（离散 + 连续）

PPO 之所以可以同时支持离散和连续动作空间，是因为它直接建模策略分布 π(a|s)。对于离散动作，可以使用 categorical 分布；对于连续动作，可以使用高斯分布。而像 DDPG 和 TD3 这类方法依赖于对动作求梯度，因此只能用于连续动作空间。

在策略梯度中对概率取对数的核心原因是利用 log-derivative trick，将概率分布的梯度转化为对数概率的梯度，从而可以将期望的梯度写成可计算的形式。虽然取对数也能带来乘法变加法和数值稳定性的好处，但这些只是附带效果，真正的关键在于使梯度表达式可优化。

### DDPG(Deep Deterministic Policy Gradient)

DDPG 是一种基于确定性策略梯度的 Actor-Critic 的 off-policy 强化学习算法，主要用于连续动作空间。它通过将策略从随机形式转为确定性函数 μ(s)，将原本对动作的期望优化转化为对 Q(s, μ(s)) 的直接优化，从而降低方差并提高效率。

在训练过程中，Critic 使用 TD 学习，通过 target network 构造稳定的目标值，而 Actor 则通过链式法则从 Critic 获取梯度更新策略。同时，DDPG 使用 replay buffer 实现 off-policy 学习，提高样本利用率。

DDPG 中 Critic 学习的是当前策略 μ 下的动作价值函数 Q^μ，通过 Bellman 方程进行 TD 更新；Actor 则在此基础上优化策略，使得 Q 值最大。主网络参数 θ 通过梯度更新，而 target 网络参数 θ′ 则通过 soft update 缓慢跟随主网络，从而提高训练稳定性

不过 DDPG 存在几个问题，比如 Q 值过估计（cirtic可能会引导actor学错方向）、对超参数敏感（actor和critic互相依赖，强耦合容易震荡），以及探索能力不足（deterministic policy必须手动加噪声），这些问题也促使后续算法如 TD3 和 SAC 的提出。

### TD3 Twin Delayed Deep Deterministic Policy Gradient

与DDPG比较类似，但是提出了几点改进：

1. Q 高估计问题（maxQ(s,a)会被选到被高估的Q）
    - **改进1（double Q- clipped double）**
    - 训练两个 Critic：$Q_1(s,a),Q_2(s,a)$
    - target变成$y=r+γmin(Q1^′(s^′,a^′),Q2^′(s^′,a^′))$
    - 取最小值 = 抑制过高估计
2. Critic 不准 → Actor 被带偏（actor依赖∇aQ(s,a)）
    - **改进2：Delayed Policy Update（延迟更新）**
    - Critic：每一步都更新, Actor：每 d 步更新一次（比如 d=2）
    - 让 Critic 更接近真实 Q，再让 Actor 学
3. target不稳定（$y=r+γQ^′(s^′,μ^′(s^′))$,  μ’可能输出极端动作）
    - **改进3：Target Policy Smoothing**
    - target 动作加噪声：$a^′=μ^′(s^′)+ϵ,  ϵ∼clip(N(0,σ),−c,c)$
    - target： $y=r+γQ^′(s^′,a^′)$
    - Actor 可能输出“极端动作”, Q 对这些点不准，本质上是不相信某一个点 → 看一个“邻域”

TD3 是对 DDPG 的改进，主要针对 Q 值过估计和训练不稳定问题。它通过引入双 Q 网络并取最小值来抑制 overestimation，通过延迟更新策略网络来避免 Actor 被不准确的 Critic 误导，同时在 target policy 上加入噪声实现平滑，从而提高训练稳定性和性能。

TD3 的核心思想可以理解为“让 Critic 更保守、更稳定”，从而保证 Actor 的更新方向是可靠的。

TD3 在 target action 上加入截断高斯噪声，从而对 Q 值进行局部平滑，避免策略过度利用 Q 函数的误差。

TD3 中的两个 Critic 网络具有相同结构但独立参数，通过随机初始化和训练过程中的随机性产生不同估计，从而在取最小值时有效抑制 Q 值过估计

### SAC(Soft actor-critic)

从目标函数层面重新定义强化学习（最大熵RL），不仅要最大化reward，还要最大化entropy (随机性）

SAC目标： $J=E[∑_tr(s_t,a_t)+αH(π(⋅∣s_t))]$

其中， $H(π(⋅∣s))=E_{a∼π}[−logπ(a∣s)]$

软bellman: $Q(s,a)=r+γE_s′[V(s′)],  V(s)=E_{a∼π}[Q(s,a)−αlogπ(a∣s)]$
也就是说：**熵项直接进入了价值函数定义**，所以后面所有更新都会带上$−αlogπ。$

构造TD target:
$y=r+γ(min(Q1^′,Q2^′)(s^′,a^′)−αlogπ(a^′∣s^′))$

Q鼓励高回报，−αlogπ：惩罚“过于确定”（低熵）。

SAC自动学习温度参数α， 目标函数是：$min_{α}E_{a∼π}[−αlogπ(a∣s)−αH_{target}]$

不能直接用Htarget 替代熵项，是因为它是一个常数，不依赖策略参数，因此对优化没有梯度贡献。SAC 中使用真实熵H(π)来影响策略更新，而Htarget则作为一个约束目标，通过调节温度参数 α 来间接控制策略的随机性。

熵项采用E[−logπ] 是因为它是信息论中唯一满足合理性质的不确定性度量，同时具有良好的可导性。SAC 的 Critic 仍然采用 MSE 形式进行回归，但目标值中引入了熵项。温度参数 α 来源于拉格朗日乘子，用于在最大化回报和维持目标熵之间进行权衡，并可以通过优化自动调整

SAC 是一种基于最大熵强化学习的 Actor-Critic 方法，它在优化回报的同时最大化策略的熵，从而将探索融入优化目标中。相比 TD3 依赖外部噪声进行探索，SAC 通过在目标函数中引入熵项，使策略在早期保持随机性，在后期逐渐收敛，从而实现更稳定和高效的学习。

### PPO（Proximal Policy Optimization）

问题出发：policy gradient的问题：$∇_θJ=E[∇logπ(a∣s)A(s,a)]$

更新一步可能太大，导致策略崩掉。理想目标是每一步**既要变好，又不能变太多**

TRPO（PPO的前身）核心思想：$max_θE[{{π_θ(a∣s)}/{π_{θ_{old}}(a∣s)}}.A(s,a)]$

同时约束：$D_{KL}(π_{old}∣∣π_θ)≤δ$   （严格限制更新步长）    问题：很难实现复杂二阶优化，不适合工程

PPO的核心思想：用一个“简单方法”近似TRPO的约束（用clip近似这个约束）

定义ratio: $r(θ)=π_θ(a∣s)/π_{old}(a∣s)$   （当前策略/旧策略采样的数据，表示新策略 vs 旧策略，对这个动作概率变化了多少;  同时需要当前/刚才的数据，所以是on-policy ）

r(θ) 是一个重要性采样比率，用来将旧策略采样的数据转换为新策略下的期望，从而保证梯度更新方向正确；而 advantage A 决定了动作是应该被强化还是削弱，两者结合可以实现正确的策略优化。

PPO的目标函数：$L=-E[min(r(θ)A,clip(r(θ),1−ϵ,1+ϵ)A)]$  clip的作用：新策略不能离旧策略太远

直觉解释：A > 0（好动作），提高概率，但不能涨太多；A < 0（坏动作），降低概率但不能降太多

PPO完整loss: $L=L_{policy}+c_1L_{value}−c_2H$

PPO 是一种基于策略梯度的 on-policy 算法，它通过引入概率比率和裁剪机制来限制策略更新幅度，从而在提升性能的同时保证训练稳定性。相比 TRPO 的复杂约束优化，PPO 用简单的 clip 操作实现了近似约束，兼顾了效果和工程可实现性。

对 policy gradient 做“trust region近似”(什意思？）**来源于KL约束，TRPO中不想让新策略离旧策略太远**

PPO 中的 Advantage 不是 Q 减去 target，而是 Q 减去 V(s)。由于 PPO 没有显式的 Q 网络，通常通过 GAE 来近似 Advantage，同时通过 A + V(s)（即近似Q(s,a)） 构造 value target 来训练 Critic。用 V + GAE 是一种更稳定的 advantage 估计方式，而不是因为 Q 不可用，而是因为在 on-policy 框架下用 Q 性价比更低。（Q比V还多依赖action，估计难度更大，on-policy 下数据利用率低，Q学不准，方差更大）

Q 函数表示在某个状态下执行某个具体动作的价值，而 V 函数表示在该状态下按照策略平均能够获得的期望回报。两者的关系是 V(s) 等于 Q(s,a) 在策略下的期望。通常 Advantage 定义为 Q(s,a) 与 V(s) 的差值，用来衡量某个动作相对于平均水平的优劣。因为动作是策略下随机采样的，会出现方差大的问题，因此需要baseline来减小方差。

MA-POCA 通常需要显式学习一个 centralized Q-function，用于评估 joint action，从而支持 counterfactual credit assignment。与 PPO 使用 GAE 基于 V 的 advantage 不同，MA-POCA 使用对动作分布求期望的 Q 作为 baseline，其目标不是单纯降低方差，而是刻画每个 agent 的边际贡献。因此，两者在设计目标上是不同的：PPO 更关注稳定性，而 MA-POCA 更关注 credit assignment，在一定程度上存在 bias-variance 的 trade-off。

### GAE Generalized Advantage Estimation（广义优势估计）

GAE 是一种计算 Advantage 的方法，用来在“偏差 vs 方差”之间做平衡

我们要用：$A(s,a)$，但问题是：A 不容易直接算

❌方法1：Monte Carlo： $A=R−V(s)$ 无偏，但方差巨大

❌方法2：TD（一步）：$δ_t=r_t+γV(s_{t+1})−V(s_t)$ 方差小，偏差大

GAE 的核心思想：把多个 **TD error 加权组合**：$A_t=∑^∞_{l=0}(γλ)^lδ_{t+l}$,其中$δ_t=r_t+γV(s_{t+1})−V(s_t)$

γ：折扣因子， λ∈[0,1]：控制 bias-variance tradeoff（λ = 0时 $A_t=δ_t$，λ = 1时，$A_t=R_t−V(s_t)$ ，相当于在 TD 和 Monte Carlo 之间做平滑过渡）

### GRPO（Group Relative Policy Optimization）

GRPO = 去掉 value function 的 PPO，用“组内相对比较”来替代 Advantage

1. 为什么需要GRPO：
    - 在 LLM 中：状态 = prompt，动作 = 整个生成序列
    - 问题1：Value function 很难学：状态空间巨大（语言），回报稀疏（只在最终给 reward），V(s) 很难准
    - 问题2：Advantage 难算，A=Q−V
    - 核心想法：不去学 V(s)，**在一组生成结果中做相对比较**

Step 1：对同一个 prompt，生成多个输出：$y_1,y_2,...,y_K$

Step 2：得到 reward, $r_1,r_2,...,r_K$（多样本比较）

Step 3：计算“相对优势”, $A_i=r_i−\frac{1}{K}∑_{j=1}^Kr_j$    (ri>平均，A>0,好；反之成立)

GRPO loss：$L=E[min(r(θ)A,clip(r(θ),1−ϵ,1+ϵ)A)]$ （和PPO一样，只有A不一样）

GRPO 是对 PPO 的一种改进，主要用于大语言模型中的强化学习场景。它不再依赖 value function 来计算 advantage，而是通过对同一输入生成多个候选输出，并基于这些输出的相对 reward 来计算 advantage，从而避免了 value function 学习困难的问题，并提高了训练稳定性。

GRPO 的本质可以理解为：**用“组内基线”替代“全局价值函数”，A=ri−baseline**

### VLM是如何guide RL 的

VLM 可以通过多种方式引导强化学习。首先，它可以作为 reward function，通过判断当前状态与语言目标的匹配程度来提供语义级奖励。其次，它可以增强状态表示，将原始视觉输入转化为高层语义特征，从而降低学习难度。最后，在更先进的框架中，VLM 还可以直接参与策略生成，作为高层规划器或条件输入，从而实现从感知到决策的端到端学习。

### Transformer

Transformer 是一种基于自注意力机制（Self-Attention）的序列建模架构，它通过建模序列中任意两个位置之间的依赖关系，替代了传统的 RNN 和 CNN，从而实现更强的并行能力和长距离依赖建模能力。

Transformer 是一种基于自注意力机制的模型，通过计算序列中各个位置之间的相关性来建模全局依赖关系。其核心是 self-attention 机制，通过 Query、Key、Value 的映射计算注意力权重，并对 Value 进行加权求和。多头注意力机制通过多个子空间的注意力并行计算，使模型能够从不同角度捕捉特征关系。相比传统的 RNN，Transformer 具有更好的并行性和长距离依赖建模能力

Transformer 之所以可以并行，是因为它用 self-attention 把序列建模从“时间递归”变成了“矩阵计算”，所有 token 之间的关系可以通过$QK^T$ 一次性算出来，而不是像 RNN 那样依赖前一步逐个计算。本质上 Transformer 把序列建模从链式依赖（sequential dependency）变成了全连接图上的一次 message passing，因此可以完全并行。

交叉熵衡量的是模型预测分布与真实分布之间的差异，在语言模型中由于真实标签是 one-hot，它等价于最小化正确词的负对数概率，从而实现最大似然训练。

### SFT（supervised fine tuning）

SFT 是监督微调，是在预训练语言模型的基础上，使用人工标注的指令-回答数据进行有监督训练，通常采用 teacher forcing 和 cross-entropy loss，使模型学会按照人类期望的方式生成回答。

预训练主要学习语言分布，而 SFT 是让模型学会“如何回答问题”，本质上是从无监督语言建模转向有监督的条件生成。

Teacher Forcing 是指在训练序列生成模型时，每一步都使用“真实的前一个 token”作为输入，而不是模型自己生成的结果。问题：Teacher forcing 会导致 exposure bias，因为训练时模型总是看到真实历史，而推理时只能依赖自己的预测，二者分布不一致。

### LoRA（Low-Rank Adaptation）

LoRA（Low-Rank Adaptation）是一种参数高效微调方法，它通过在原模型权重上引入低秩矩阵分解的增量更新，在冻结原模型参数的情况下，只训练少量新增参数，从而大幅降低训练成本。

为什么low rank是合理的：LoRA（Low-Rank Adaptation）是一种参数高效微调方法，它通过在原模型权重上引入低秩矩阵分解的增量更新，在冻结原模型参数的情况下，只训练少量新增参数，从而大幅降低训练成本。

r 是低秩分解的秩，控制 LoRA 的表达能力。参数量是 r×d + d×r，比如 d=4096，r=8，总共约 65K，相比原来的 16M（d×d） 大幅减少。

Transformer 在预训练阶段是全量训练的，SFT 和 LoRA 是在预训练完成后，对大模型进行微调时使用的方法，其中 SFT 是目标，LoRA 是参数高效实现方式。

预训练属于自监督学习，通过 next-token prediction 从文本中自动构造标签，因此既不需要人工标注，又可以用监督学习的方式训练。

### RLHF（Reinforcement Learning from Human Feedback）

RLHF（Reinforcement Learning from Human Feedback）是通过训练一个奖励模型来模拟人类偏好，并用强化学习（通常是 PPO）优化语言模型，使其生成更符合人类期望的输出。

整体流程：

<aside>
💡

> Pretraining（预训练）
↓
SFT（学会回答）
↓
Reward Model（学人类偏好）
↓
PPO（优化输出）
> 
</aside>

PPO 用于优化语言模型这个 policy，使其生成更高 reward 的文本，同时通过 clipping 和 KL penalty 限制更新幅度，防止模型偏离原始分布过多。

RLHF在PPO的基础上加入了KL惩罚：

$R=R_{RM}−β⋅KL(π∣∣π_{ref})$    $R_{RM}(x, y) =$ 奖励模型给这个回答的评分

$KL(π∣∣π_{ref})=∑πlog\frac{π}{π_{ref}}$

PPO 的 clip 只约束每一步更新中 π_new 和 π_old 的差异，是一种局部约束；而 RLHF 中的 KL 惩罚约束的是当前策略与参考模型之间的偏离，是一种全局约束。由于多次更新后策略可能逐渐偏离原始语言分布，因此需要引入 KL 项来防止 reward hacking 并保持语言能力。

在 RLHF 中，KL 惩罚是通过 reward shaping 的方式加入的，即将 reward 修改为$R=R_{RM}−β⋅KL(π∣∣π_{ref})$, 然后使用这个新的 reward 来计算 advantage，再进入 PPO 的优化过程，而不是直接加到 PPO 的 loss 上

TRPO 中的 KL 是作为硬约束出现的，需要解带约束优化问题，因此计算复杂；而 RLHF 中的 KL 是作为一个 penalty 项加入到 reward 中，是软约束形式，可以通过 log probability 的差值高效计算，因此在工程上更可行。

### DPO （Direct Preference Optimization）

DPO 是一种不使用强化学习的方法，它直接利用人类偏好数据进行训练，通过最大化优选回答相对于劣质回答的概率差，从而对齐模型输出。PPO 需要先训练 reward model，再用强化学习优化策略，而 DPO 直接将 preference learning 转化为监督学习问题，避免了 RL 的不稳定性和复杂性

DPO 是在 SFT 模型基础上，直接用 preference 数据构造一个对比 loss（基于 log probability 差），通过监督学习方式优化模型。

$$
loss = - log σ(β (log π(y_w|x) - log π(y_l|x)))
$$

在 RLHF 中，最优策略满足

$log⁡π(y∣x)=log⁡π_{ref}(y∣x)+1βR(y)$

因此 reward 可以用 log probability 差来表示。DPO 正是利用这一关系，用$log⁡π(y_w∣x)−log⁡π(y_l∣x)$来替代 reward 差，从而避免显式训练 reward model。

### Diffusion policy

把“生成动作序列”当成一个去噪过程来做。也就是说，从随机噪声动作开始，一步步去噪 → 最终生成合理的动作。**就是把Diffusion Model 用在 action 上**

Diffusion Policy 是一种基于 diffusion model 的策略学习方法，它将动作生成建模为一个逐步去噪的过程。在训练阶段，我们对 expert action 加噪，并训练一个网络去预测噪声；在推理阶段，从随机噪声开始，通过多步去噪生成动作序列。相比传统 behavior cloning，它可以更好地建模 multi-modal action 分布，并且能够生成一段 trajectory，从而具备一定的规划能力。同时它的训练是类似监督学习的，因此比 reinforcement learning 更稳定、更高效。

### diffusion model

Diffusion model 将生成建模为一个逐步去噪过程，通过训练模型预测不同噪声水平下的噪声，从而在推理时从纯噪声逐步还原出真实数据。

1️⃣ Forward process（加噪）

从真实动作$a_0$开始，逐步加噪：$a_t=\sqrt{a_t}a_{t-1}+\sqrt{(1-\alpha_t)\epsilon}$

最终变成纯噪声

2️⃣ Reverse process（去噪）

训练一个网络：$ϵ_θ(a_t,o,t)$, 作用是预测当前噪声，然后一步步去噪：$a_{t-1}=f(a_t,\epsilon_\theta)$