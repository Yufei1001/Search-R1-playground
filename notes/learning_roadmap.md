# Agentic RL 学习路线：从 Search-R1 出发

2026-11-09 · Yufei1001

## 总览

用大约 10 周，以 Search-R1 为主线，从 RL 原理一路走到能独立设计并训练一个新的 agentic RL 任务。原则是"读一篇、推一遍、跑一次"：每个概念都要落到公式、代码行和一条训练曲线上。

**最终要能做到的事：**

- 讲清楚 PPO / GRPO 在 LLM 多轮工具调用场景下的每一项 loss，并能手推。
- 在 8×A800 上独立复现 Search-R1（Qwen2.5-3B/7B），复现 NQ / HotpotQA 上的 EM 结果。
- 读懂 veRL 的训练循环，知道每个超参数影响哪块显存、哪段耗时、哪个指标。
- 能诊断常见故障：reward hacking、熵坍缩、回复长度爆炸、KL 发散、训练突然崩。
- 设计一个新环境（新工具或新任务），从零训练出一个 agent。

**五个阶段：**

1. **基础（第 1–2 周）**：RL 基础 + RLHF/PPO，读 InstructGPT、PPO、GAE。
2. **推理 RL（第 3 周）**：GRPO、DeepSeek-R1，以及 DAPO、Dr. GRPO 等改进。
3. **Search-R1 复现（第 4–6 周）**：精读代码，跑通检索服务和训练，复现论文结果。
4. **消融与诊断（第 7–8 周）**：改一个变量跑一组，建立对训练动态的直觉。
5. **扩展（第 9–10 周）**：换工具、换任务或换框架，做一个属于自己的小项目。

**每个阶段的产出物**放在 repo 的 `notes/`（读书笔记、推导）和 `docs/experiment_log.md`（实验记录）里，通过 git 在 Mac 和开发机之间同步。

## 前置基础：RL 与 LLM 上的 RL

这一步的目标是把"语言模型生成"翻译成 RL 的语言，之后所有论文都建立在这套对应关系上。不需要学完整的经典 RL（Q-learning、model-based 等可以跳过），只需要策略梯度这一支。

**LLM 即策略：一张对照表**

| RL 概念 | 单轮文本生成 | Agentic（多轮工具调用） |
| --- | --- | --- |
| 状态 s_t | prompt + 已生成的 token | prompt + 模型 token + 工具返回的 observation |
| 动作 a_t | 下一个 token | 下一个 token（一段连续 token 组成一次工具调用） |
| 策略 π_θ | LLM 的 softmax 分布 | 同左 |
| 环境转移 | 确定性：拼接 token | 碰到 `</search>` 时由外部工具拼入结果，不可微、有随机性 |
| 奖励 r | 通常只在序列末尾给一次 | 同左，可加中间奖励（格式、检索命中） |
| 轨迹 τ | 一条 response | 一条多轮交互记录（模型与工具交替） |

关键差别：agentic 轨迹里有一部分 token 不是策略生成的。它们要参与前向（作为上下文），但不能参与 loss，这就是 Search-R1 里的 retrieved token masking。

**要掌握的概念（按顺序）**

1. MDP、回报、价值函数 V(s) 与 Q(s,a)、优势 A = Q − V。
2. 策略梯度定理与 REINFORCE：为什么梯度是 ∇logπ × 回报。
3. Baseline 与方差：减一个与动作无关的量不改变期望，但能降方差。GRPO 的组均值就是一个 baseline。
4. Actor-Critic 与 GAE：用学出来的 V 估计优势，λ 控制偏差与方差的折中。
5. 重要性采样与 off-policy：一批数据更新多次时，为什么要乘以 π_θ / π_old。
6. TRPO → PPO：用 clip 近似信赖域，限制每步更新幅度。
7. KL 正则：限制策略偏离参考模型，防止语言能力退化和 reward hacking。
8. 熵：策略的探索能力。熵坍缩是 LLM RL 最常见的失败模式之一。

**推荐材料**

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)：Part 1–3，重点看策略梯度推导，1–2 天。
- Lilian Weng，[Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)：一篇文章讲完从 REINFORCE 到 PPO 的脉络。
- Sergey Levine CS285 第 5–9 讲（策略梯度、Actor-Critic、高级 PG）：想把数学吃透的话看。
- [The N+ Implementation Details of RLHF with PPO](https://iclr-blogposts.github.io/2024/blog/the-n-implementation-details-of-rlhf-with-ppo/)：工程细节的最佳入门，读完再看 veRL 会顺很多。

**自检**：空手推导 REINFORCE 梯度；说清楚为什么 LLM RL 里的 γ 通常取 1；解释为什么奖励只在最后一个 token 上，但前面每个 token 都能拿到梯度。

## 必读论文清单

按顺序读约 20 篇，标★的精读（推公式、对代码），其余略读（弄懂动机和改动点即可）。链接的 arXiv 编号是凭记忆写的，打不开就按标题搜索。

| 阶段 | 论文 | 读法 | 要带走的东西 |
| --- | --- | --- | --- |
| 基础 | ★ [PPO](https://arxiv.org/abs/1707.06347)（Schulman 2017） | 精读 | clip 目标、为什么可以多 epoch 复用数据 |
| 基础 | ★ [GAE](https://arxiv.org/abs/1506.02438)（Schulman 2015） | 精读第 3–4 节 | δ_t、λ 加权、偏差方差权衡 |
| RLHF | ★ [InstructGPT](https://arxiv.org/abs/2203.02155) | 精读方法部分 | SFT → RM → PPO 流程、KL 惩罚的位置 |
| RLHF | [DPO](https://arxiv.org/abs/2305.18290) | 略读 | 理解"不用 RL 也行"的路线，作为对照 |
| RLHF | [RLOO / Back to Basics](https://arxiv.org/abs/2402.14740) | 略读 | 去掉 critic、用组内其他样本做 baseline |
| 推理 RL | ★ [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300) | 精读第 4 节 | 组归一化优势、KL 放进 loss 而非 reward |
| 推理 RL | ★ [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 精读 R1-Zero 部分 | 纯规则奖励 + GRPO 产生长思维链 |
| 推理 RL | [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 略读 | 长度惩罚、长上下文 RL |
| 算法改进 | ★ [DAPO](https://arxiv.org/abs/2503.14476) | 精读 | clip-higher、动态采样、token 级 loss、超长惩罚 |
| 算法改进 | ★ [Dr. GRPO](https://arxiv.org/abs/2503.20783) | 精读 | GRPO 的长度偏差和标准差偏差 |
| 算法改进 | [REINFORCE++](https://arxiv.org/abs/2501.03262)、[VAPO](https://arxiv.org/abs/2504.05118) | 略读 | 无 critic 与有 critic 两条路的改进 |
| 算法改进 | [Entropy Mechanism of RL for Reasoning LMs](https://arxiv.org/abs/2505.22617) | 略读 | 熵坍缩的成因与对策 |
| 算法改进 | [GSPO](https://arxiv.org/abs/2507.18071) | 略读 | 序列级重要性比，MoE 上更稳 |
| Agent 前史 | [ReAct](https://arxiv.org/abs/2210.03629)、[Toolformer](https://arxiv.org/abs/2302.04761) | 略读 | 思考-行动-观察的格式从哪来 |
| 搜索 Agent | ★ [Search-R1](https://arxiv.org/abs/2503.09516) | 精读 + 对代码 | 多轮 rollout、检索 token mask、EM 奖励 |
| 搜索 Agent | ★ [Search-R1 实证研究](https://arxiv.org/abs/2505.15117) | 精读 | 格式奖励、中间奖励、模型与检索器的影响 |
| 搜索 Agent | [R1-Searcher](https://arxiv.org/abs/2503.05592)、[ReSearch](https://arxiv.org/abs/2503.19470) | 略读 | 同期工作的设计差异 |
| 搜索 Agent | [DeepResearcher](https://arxiv.org/abs/2504.03160)、[ZeroSearch](https://arxiv.org/abs/2505.04588) | 略读 | 真实网络环境 vs. 用 LLM 模拟搜索 |
| 通用 Agent | [ToRL](https://arxiv.org/abs/2503.23383)、[ToolRL](https://arxiv.org/abs/2504.13958) | 略读 | 代码解释器与通用工具的奖励设计 |
| 通用 Agent | ★ [RAGEN](https://arxiv.org/abs/2504.20073) | 精读 | 多轮 RL 的"回音陷阱"与轨迹级优化 |
| 通用 Agent | [SWE-RL](https://arxiv.org/abs/2502.18449) | 略读 | 软件工程任务上的 RL |
| 系统 | ★ [HybridFlow（veRL）](https://arxiv.org/abs/2409.19256) | 精读第 3–5 节 | 单控制器、actor/rollout 共置、重分片 |

**补充博客**：John Schulman 的 [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)（k1/k2/k3 估计器，必读）；Lilian Weng 的 [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)。

**读法**：每篇精读在 `notes/papers/` 留一个文件，写三件事：解决什么问题、核心公式（手抄）、对应 veRL 里哪几行代码或哪个配置项。对不上代码的地方标记为问题，留给消融实验。

## 核心数学推导

下面 7 组公式是整条路线的数学骨架，每组都标了在本 repo 里的实现位置。建议每组自己在纸上推一遍，再去对代码。

### 1. 策略梯度定理

从 J(θ) = E_τ[R(τ)] 出发，用 log-derivative trick（∇p = p∇log p）得到：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_t\Big]
$$

要理解的点：环境转移概率不含 θ，所以导数里消掉了——这就是为什么搜索引擎不可微也不影响训练。把回报换成优势 A_t（减去 baseline）不改变期望，只降方差。

### 2. PPO-clip 目标

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, A_t\big)\Big], \quad \rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

A > 0 时 ρ 最多涨到 1+ε，A < 0 时最多降到 1−ε，超出部分梯度为 0。对应 `verl/trainer/ppo/core_algos.py` 的 `compute_policy_loss`，监控指标 `pg_clipfrac` 就是被截断的 token 占比。

### 3. GAE

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l\, \delta_{t+l}
$$

λ = 0 就是单步 TD（低方差高偏差），λ = 1 就是蒙特卡洛回报减 V（高方差无偏）。LLM RL 里通常 γ = λ = 1。对应 `compute_gae_advantage_return`；想一想：被 mask 掉的检索 token 在 GAE 的递推里应该怎么处理？

### 4. PPO 中的 KL 奖励整形

veRL 的 PPO 路径把 KL 放进奖励，而不是 loss：

$$
r_t = R(\tau)\cdot \mathbb{1}[t = T] \;-\; \beta \big(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t)\big)
$$

`train_ppo.sh` 里的 `algorithm.kl_ctrl.kl_coef=0.001` 就是 β。对应 `compute_rewards` 和 `kl_penalty`。

### 5. GRPO 优势与 KL 估计器

同一个 prompt 采 G 条回复，用组内统计量做 baseline，不需要 critic：

$$
\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \varepsilon}
$$

KL 直接加在 loss 里，用 Schulman 的 k3 估计器（无偏、恒非负、方差小）：

$$
\mathbb{D}_{\text{KL}}^{k3} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1
$$

对应 `compute_grpo_outcome_advantage`（注意它对组内只有一条样本时把 std 设为 1）。一个必须想明白的问题：一组全对或全错时优势全为 0，这批数据就白跑了——DAPO 的动态采样就是为此而来。

### 6. Loss 的聚合方式

同样的 per-token loss ℓ_{i,t}，不同的平均方式会给长短回复不同权重：

- GRPO 原版（样本级）：

$$
\frac{1}{G}\sum_{i}\frac{1}{|o_i|}\sum_{t}\ell_{i,t}
$$

- DAPO（token 级）：

$$
\frac{1}{\sum_i |o_i|}\sum_{i}\sum_{t}\ell_{i,t}
$$

- Dr. GRPO（常数归一）：

$$
\frac{1}{G \cdot L_{\max}}\sum_{i}\sum_{t}\ell_{i,t}
$$

样本级平均会让长的错误回复每个 token 受罚更轻，导致回复越写越长。本 repo 的 `compute_policy_loss` 用 `masked_mean` 对整个 micro batch 做 token 级平均。

### 7. 检索 token mask

Agentic RL 特有的一步。设 m_t = 0 表示第 t 个 token 来自 `<information>` 块（工具返回），否则为 1：

$$
L = \frac{\sum_{t} m_t\, \ell_t}{\sum_t m_t}, \qquad m_t = \mathbb{1}[\,a_t \text{ 由 } \pi_\theta \text{ 生成}\,]
$$

不加 mask 时，模型会被要求"去拟合"检索到的文档，这些 token 在 π_θ 下概率很低，重要性比和梯度都会很奇怪。实验日志的 v0.2 说明修复这个 bug 后训练稳定性大幅提升。对应 `generation.py` 的 `info_mask` 和 `dp_actor.py` 的 `state_masking` 分支。

## Search-R1 代码精读路线

整个项目只有一个新东西：把 veRL 的单次生成换成了"生成 → 搜索 → 拼接结果 → 再生成"的多轮循环，其余都是标准 PPO/GRPO。按下面顺序读，每一步带着问题去看。

| 顺序 | 文件 | 看什么 | 读完要能回答 |
| --- | --- | --- | --- |
| 1 | `scripts/data_process/nq_search.py` | prompt 模板、`<think>` / `<search>` / `<answer>` 标签约定 | 模型从哪里学到要用这些标签？base 模型和 instruct 模型有什么区别？ |
| 2 | `search_r1/search/retrieval_server.py` | E5 + FAISS 检索服务，FastAPI 接口 `/retrieve` | 一次检索延迟多少？为什么要单独起服务而不是在训练进程里查？ |
| 3 | `search_r1/llm_agent/generation.py` 的 `run_llm_loop` | 多轮循环、`active_mask`、`max_turns` | 一个 batch 里有的轨迹已经答完，有的还在搜，怎么处理？ |
| 4 | 同上的 `execute_predictions`、`postprocess_predictions` | 解析 `<search>` / `<answer>`，非法动作的提示语 | 模型输出格式错了会怎样？这算不算一种环境反馈？ |
| 5 | 同上的 `_info_masked_concatenate_with_padding`、`_compose_final_output` | `info_mask` 的构造 | 哪些 token 进入 attention、哪些进入 loss？ |
| 6 | `verl/trainer/ppo/ray_trainer.py` 的 `fit` | 一个训练步的完整顺序：rollout → old/ref log-prob → value → reward → advantage → update | 每一步在哪组 GPU 上、哪个模型上计算？ |
| 7 | `verl/trainer/main_ppo.py` 的 `RewardManager` + `verl/utils/reward_score/qa_em.py` | EM 奖励、`normalize_answer` | 奖励放在哪个 token 上？怎么容易被 hack？ |
| 8 | `verl/trainer/ppo/core_algos.py` | GAE、GRPO 优势、policy loss、value loss、KL | 把上一节的公式一一对应到代码行 |
| 9 | `verl/workers/actor/dp_actor.py` 的 `update_policy` | mini/micro batch、梯度累积、`state_masking` | `ppo_mini_batch_size` 和 `ppo_micro_batch_size` 各自影响什么？ |
| 10 | `verl/workers/fsdp_workers.py` + `verl/single_controller/` | FSDP 与 vLLM 的权重同步、Ray worker group | 为什么 actor 和 rollout 共用同一组 GPU？ |

**读代码的方法**

- 先在开发机上用 1 张卡、Qwen2.5-0.5B、`train_batch_size=8` 跑通，然后在 `run_llm_loop` 和 `update_policy` 里打断点或加打印，看每个 tensor 的 shape。
- 打印一条完整轨迹：解码 `input_ids`，把 `info_mask` = 0 的位置标红，确认只有 `<information>` 内容被 mask。
- 读 `infer.py` 对照：它是同一个循环的简化版，没有训练相关的 padding 和 batch 处理，先读它更容易理解主循环。

这个 repo 的 veRL 是 2025 年初的旧版。新版 veRL 已经内置了多轮 rollout（`multi_turn` 配置、SGLang/vLLM 的 agent loop），理解这里的实现之后再去读新版，会很清楚它们在解决什么问题。

## RL 训练细节与调参

veRL 的一个训练步分成两段：先用 vLLM 做 rollout（生成 + 搜索），再把同一份权重切回 FSDP 做训练。在 Search-R1 上，大部分时间花在 rollout，所以吞吐优化优先看生成和检索。

### veRL 架构要点

- **Single controller**：一个驱动进程（`RayPPOTrainer.fit`）按顺序调度各个 worker group，算法逻辑全在这一处，改算法只需要改这里和 `core_algos.py`。
- **Hybrid engine**：actor 和 rollout 共用同一组 GPU。rollout 时权重从 FSDP 分片重组为 vLLM 的 TP 分片，训练时再释放 KV cache 切回来（`sharding_manager/`）。
- **四个角色**：actor（训练的策略）、rollout（vLLM 推理）、ref（冻结的初始模型，算 KL）、critic（只有 PPO 需要）。GRPO 省掉 critic，显存和时间都少一大块。
- **DataProto**：各角色之间传递的数据容器，一个训练步里它会逐步长出 `old_log_probs`、`ref_log_prob`、`values`、`token_level_rewards`、`advantages` 等字段。

### 关键超参数（train_grpo.sh / train_ppo.sh 的默认值）

| 参数 | 默认值 | 作用 | 调大会怎样 |
| --- | --- | --- | --- |
| `data.train_batch_size` | 512 | 每步采样的 prompt 数 | 梯度更稳，每步更慢 |
| `rollout.n_agent` | GRPO 5 / PPO 1 | 每个 prompt 采几条轨迹（GRPO 的 G） | 组内 baseline 更准，rollout 成本线性增加 |
| `actor.ppo_mini_batch_size` | 256 | 一次优化器更新用多少样本 | 每批数据更新次数变少，更 on-policy |
| `actor.ppo_micro_batch_size` | 64 | 单次前向的样本数，只影响显存 | OOM；不改变数学结果 |
| `actor.optim.lr` | 1e-6 | actor 学习率 | 快但容易崩，LLM RL 一般在 1e-7 到 3e-6 |
| `lr_warmup_steps_ratio` | 0.285 | warmup 占总步数的比例 | 开局更稳，前期学得慢 |
| `kl_coef` / `kl_loss_coef` | 0.001 | KL 系数 β | 策略被拴在 ref 附近，学不动 |
| `rollout.temperature` | 1 | 采样温度 | 探索更多，格式更容易乱 |
| `max_turns` | 2 | 最多搜几轮 | 能做多跳问题，序列更长 |
| `retriever.topk` | 3 | 每次检索返回几篇 | 信息更全，上下文更长 |
| `rollout.gpu_memory_utilization` | 0.6 | vLLM 可用显存比例 | rollout 更快，训练阶段可能 OOM |
| `state_masking` | true | 是否 mask 检索 token | 关掉就是消融实验里的对照组 |

脚本里的注释给了 `max_prompt_length` 的计算公式：`max_start_length + max_response_length × (max_turns − 1) + max_obs_length × max_turns`。改 `max_turns` 时要同步算这个值。

### 要监控的指标

| 指标 | 正常表现 | 异常信号 |
| --- | --- | --- |
| `critic/score/mean`（训练奖励） | 平稳上升后趋平 | 突然陡降 = 训练崩了 |
| `val/test_score` | 跟训练奖励同向 | 训练涨、验证不涨 = 过拟合或 reward hacking |
| `actor/entropy_loss`（熵） | 缓慢下降 | 快速跌到接近 0 = 熵坍缩，不再探索 |
| `actor/kl_loss` 或 `actor/reward_kl_penalty` | 缓慢增长 | 指数级增长 = 策略在发散 |
| `actor/pg_clipfrac` | 约 0.001–0.05 | 持续偏高 = 学习率过大或更新太 off-policy |
| `response_length/mean` | 先调整后稳定 | 一路涨到上限 = 长度偏差或重复生成 |
| `env/number_of_valid_search`、`env/finish_ratio` | 搜索次数稳定在 1–2 | 降到 0 = 模型学会了不搜索直接猜 |
| `critic/vf_explained_var`（PPO） | 逐步上升到正值 | 长期为负 = critic 没学好，优势估计都是噪声 |

指标名以你自己的 wandb 为准，第一次跑时把实际名字核对到这张表里。

### 常见故障与排查

- **训练崩溃**（奖励突然归零、输出乱码）：先查检索 token mask 是否生效，再降学习率、加大 KL。Search-R1 实证研究发现 PPO 比 GRPO 更不容易崩。
- **熵坍缩**：用 DAPO 的 clip-higher（上界 ε 大于下界）、提高温度或加小的熵奖励。
- **Reward hacking**：EM 奖励下模型可能学会在 `<answer>` 里堆多个候选。每 50 步抽 20 条轨迹人工看。
- **回复越来越长**：换 token 级 loss 聚合、去掉 std 归一（Dr. GRPO）、加超长惩罚。
- **OOM**：先降 `ppo_micro_batch_size` 和 `log_prob_micro_batch_size`，再降 `gpu_memory_utilization`，最后开 offload（默认脚本已开，会变慢）。
- **复现性**：固定 seed，记录 git commit、完整命令行和检索器版本。RL 单次运行方差很大，关键对比至少跑 2 个 seed。

## 8×A800 实验计划

先用 3B 模型把流程和消融都跑完，最后再上 7B。一次只改一个变量，每个实验开始前写下预期结果，跑完对照预期写结论。耗时依赖检索速度和序列长度，第一个完整训练跑完后再按实测时间排期。

### 环境准备（第 4 周）

1. 按 README 在开发机上建两个 conda 环境：训练用的 `searchr1`（torch + vLLM + veRL）和检索用的 `retriever`（faiss-gpu）。两者分开是为了避免依赖冲突。
2. 下载 wiki-18 语料和 E5 索引（`scripts/download.py`），准备 NQ + HotpotQA 数据（`scripts/nq_hotpotqa/data_process.sh`）。
3. 启动检索服务，用 `search_r1/search/retrieval_request.py` 测一次请求，记录单次延迟和显存占用。
4. 冒烟测试：Qwen2.5-0.5B、`train_batch_size=16`、`total_training_steps=5`，确认整个链路能跑完、wandb 有曲线。

**注意 GPU 分配**：脚本默认训练占满 8 张卡，检索服务的 `--faiss_gpu` 也要占显存。如果 OOM，可以让检索在 CPU 上跑，或者分出 1 张卡给检索、用 7 张训练（要同步改 `n_gpus_per_node` 和 batch 参数，保证能整除）。

### 实验列表

| # | 实验 | 改什么 | 想弄懂的问题 |
| --- | --- | --- | --- |
| E0 | 无训练基线 | 只跑验证：直接回答 / 标准 RAG / 未训练模型 + 搜索模板 | RL 到底带来了多少提升？ |
| E1 | ★ 复现 GRPO | `train_grpo.sh`，Qwen2.5-3B-Base | 能否接近论文结果？记录每步耗时和各阶段占比 |
| E2 | ★ 复现 PPO | `train_ppo.sh`，同一模型 | PPO 与 GRPO 在收敛速度、稳定性、最终分数上的差异 |
| E3 | 关掉检索 mask | `state_masking=false` | 亲眼看到 mask 的作用：曲线有多不稳？ |
| E4 | Base vs. Instruct | 换 Qwen2.5-3B-Instruct | 初始格式遵循能力对学习曲线的影响 |
| E5 | 组大小 G | `n_agent` = 3 / 5 / 8 | baseline 精度与 rollout 成本怎么折中 |
| E6 | KL 系数 | `kl_loss_coef` = 0 / 0.001 / 0.01 | KL 是在保护还是在拖后腿 |
| E7 | 格式奖励 | `main_ppo_format.py` + `scripts/nq_hotpotqa/v0.3/` | 格式奖励对 base 模型是否更重要 |
| E8 | 搜索轮数 | `max_turns` = 1 / 2 / 4 | 多跳数据集（HotpotQA、2Wiki）是否受益更多 |
| E9 | 换检索器 | E5 换成 BM25（`example/retriever/`） | 检索质量弱时，策略会学出什么不同行为 |
| E10 | 自己实现一个算法改进 | 在 `core_algos.py` 里加 clip-higher 或 Dr. GRPO（去 std） | 能否改善熵和回复长度的问题 |
| E11 | 上 7B | 最好的配置换 Qwen2.5-7B | 模型变大后的收益和成本 |

**评测**：用 `scripts/nq_hotpotqa/evaluate.sh` 在 7 个数据集上算 EM，每个实验在 `docs/experiment_log.md` 里记一行：日期、commit、配置差异、wandb 链接、各数据集 EM、一句话结论。

## Mac ↔ 开发机 git 协作工作流

Mac 负责读代码、写笔记、改脚本，开发机只负责跑实验；git 只同步代码和文字，数据、checkpoint、日志留在开发机，曲线走 wandb。

**一次性设置**

1. 开发机上生成一把自己的 SSH key 并加到 GitHub（不要把 Mac 的私钥拷过去），然后 clone 同一个 repo。
2. 在 repo 里加 `.gitignore`：`data/`、`verl_checkpoints/`、`wandb/`、`outputs/`、`*.log`、`*.parquet`、`*.index`。
3. 开发机上 `wandb login`，所有实验记在同一个 project 下，Mac 上直接在浏览器看曲线。
4. 长任务放在 `tmux` 里跑，SSH 断开不影响训练。

**日常循环**

1. Mac：从 `main` 拉一个实验分支，如 `exp/e3-no-mask`，改脚本，提交并 push。
2. 开发机：`git fetch && git checkout exp/e3-no-mask`，在 tmux 里启动训练。
3. 跑完后在开发机上把评测结果写进 `docs/experiment_log.md`，提交并 push；不要在开发机上改代码，避免两边冲突。
4. Mac：`git pull`，写分析笔记；结论稳定的改动（如新算法）merge 回 `main`。

**建议的目录**

- `notes/papers/`：每篇论文一个文件。
- `notes/derivations/`：手推公式（可以拍照或用 Markdown + LaTeX）。
- `experiments/<实验名>.sh`：每个实验一个独立脚本，脚本就是配置的记录，不在命令行临时改参数。
- `docs/experiment_log.md`：所有实验的结果表。

要让 Claude Code 协助的话，在 Mac 上让它帮你写实验脚本、分析从开发机拷回来的日志片段或 wandb 导出的 CSV，或者在开发机上也装一个 Claude Code 直接排查训练问题。

## Agentic RL 其他必备

从搜索扩展到通用 agent 时，难点从"算法"转向"环境、奖励和系统"。下面 6 个主题是 Search-R1 里不太显眼、但做自己的项目时一定会碰到的。

### 1. 环境与工具设计

- 动作格式：自定义标签（`<search>`）还是模型自带的 function calling 格式？后者对 instruct 模型更友好。
- Observation 的长度和截断规则（`max_obs_length`）直接决定上下文增长速度。
- 非法动作要有明确反馈（就像 `execute_predictions` 里那句 "Let me try again"），否则模型不知道自己错在哪。
- 工具要可复现、可并发、低延迟。真实 API（网页搜索）有费用、限流和结果漂移，训练时常用本地索引或缓存替代。

### 2. 奖励设计

- 结果奖励（EM、F1、单元测试通过）稳定但稀疏；过程奖励（检索命中、格式）密集但容易被 hack。Search-R1 实证研究发现格式奖励有用，中间检索奖励收益有限。
- 开放式任务（写报告、长答案）没有标准答案，需要 LLM-as-judge 或奖励模型，此时要特别提防策略去讨好评委。
- 可验证奖励（RLVR）是目前最可靠的路线：选任务时优先选能自动判对错的。

### 3. 多轮信用分配

- 只有最后一个答案有奖励时，哪一次搜索起了作用？GRPO 给整条轨迹同一个优势，PPO 的 critic 理论上能分到每一步。
- 轮数变多（10 轮以上）后，信用分配变难，可以看 RAGEN、verl-agent 里的 turn-level / step-level 优势。

### 4. 系统：长尾与异步

- 同步 rollout 里，一个 batch 要等最慢的轨迹跑完（长尾），工具调用让这个问题更严重。
- 异步 RL：rollout 和训练分到不同 GPU 上并行，代价是数据有一定 off-policy 程度，需要重要性采样修正。AReaL、slime 是这个方向的代表。
- 长上下文：多轮轨迹动辄几万 token，需要序列并行（Ulysses）、动态 batch（`use_dynamic_bsz`）和上下文管理策略。

### 5. 训练/推理不一致

vLLM 采样时的概率和 FSDP 重算的概率并不完全一致（精度、kernel 差异），在长序列和 MoE 模型上会导致训练不稳。当成一个已知风险，遇到无法解释的崩溃时记得查。

### 6. 框架生态

| 框架 | 定位 | 什么时候看 |
| --- | --- | --- |
| [veRL](https://github.com/volcengine/verl) | 字节跳动，当前最主流的 LLM RL 框架 | 主线；复现完 Search-R1 后读新版的多轮 agent loop |
| [verl-agent](https://github.com/langfengQ/verl-agent) | 基于 veRL 的多轮 agent 扩展 | 做长 horizon 任务（游戏、网页）时 |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN) | 多轮 agent RL 研究框架 | 研究多轮稳定性时 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM，代码较简洁 | 想看另一种实现对照时 |
| [TRL](https://github.com/huggingface/trl) | Hugging Face，单机易用 | 小模型快速验证想法时 |
| [AReaL](https://github.com/inclusionAI/AReaL)、[slime](https://github.com/THUDM/slime) | 异步 / 大规模 RL | 学异步训练时 |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL)、[rLLM](https://github.com/agentica-project/rllm) | 面向 agent（如 SWE 任务）的 RL | 做代码 agent 时 |

## 时间表与自检清单

按每周大约 15 小时估算，共 10 周。过不了关就先别进入下一阶段；提前过关就把时间挪给第 4–8 周的实验。

| 阶段 | 周次 | 内容 | 过关标准 |
| --- | --- | --- | --- |
| 基础 | 第 1–2 周 | RL 基础、PPO、GAE | 能手推 PG / PPO |
| 推理 RL | 第 3 周 | GRPO、R1、DAPO、Dr. GRPO | 能讲清 GRPO 优势 |
| **复现** | 第 4–6 周 | 环境 + 检索服务，E0–E2 | EM 接近论文 |
| 消融与诊断 | 第 7–8 周 | E3–E10，一次改一个变量 | 每个实验有结论 |
| 扩展 | 第 9–10 周 | 新任务或新工具，E11 上 7B | 新环境训练有效 |

第 4 周开始用 GPU，第 1–3 周可以提前把开发机环境和数据下载做好，等读完论文就能直接开跑。

### 自检清单

- [ ] 空手推导策略梯度定理，解释为什么环境（搜索引擎）不可微也能训练
- [ ] 画出 PPO-clip 目标在 A > 0 和 A < 0 时随 ρ 变化的曲线
- [ ] 说清 PPO 把 KL 放在 reward 里、GRPO 放在 loss 里的区别，以及 k3 估计器为什么非负
- [ ] 解释样本级和 token 级 loss 聚合对回复长度的影响
- [ ] 指出 `core_algos.py` 里每个公式对应的行
- [ ] 打印一条训练轨迹，标出哪些 token 被 `info_mask` 排除在 loss 外
- [ ] 完成 E1 和 E2，NQ / HotpotQA 上的 EM 与论文报告值对比并解释差距
- [ ] 看一条 wandb 曲线就能判断是熵坍缩、KL 发散还是长度爆炸
- [ ] 在 `core_algos.py` 里自己实现一个算法改进，并用实验证明或否定它
- [ ] 接入一个新工具或新任务，写出奖励函数，训练后验证集分数高于未训练基线
