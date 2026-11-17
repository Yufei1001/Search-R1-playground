# Agentic RL 实验计划：8 卡 / 4 卡两套配置

2026-11-17 · Yufei1001

本文是 `learning_roadmap.md` 里实验列表（E0–E11）的落地版：每个实验写清楚改哪个参数、在哪种硬件配置下跑、看哪几个指标、预期是什么。原则还是**一次只改一个变量，只和同一配置下的基线比**。

## 0. 开跑前先弄清楚的四件事

这几条是读代码时确认过的，直接影响实验设计。

**① batch 大小与卡数无关，只影响每张卡上分到多少。**
`fsdp_workers.py:97` 会把 `ppo_mini_batch_size`、`ppo_micro_batch_size`、`log_prob_micro_batch_size` 除以卡数，所以脚本里写的都是**全局值**。从 8 卡换到 4 卡、全局值不变时，数学上是同一个实验，只是每卡显存翻倍、耗时变长。

**② GRPO 默认配置每步会做 10 次参数更新。**
`ray_trainer.py:702` 按 `n_agent` 复制 batch，但 `fsdp_workers.py:100` 乘的是 `rollout.n`（默认为 1），不是 `n_agent`。所以 `train_batch_size=512, n_agent=5` 产生 2560 条轨迹，而 `ppo_mini_batch_size=256` 指的是 256 条**轨迹**，结果是每步更新 2560 / 256 = 10 次。后面几次更新已经明显 off-policy，`pg_clipfrac` 会比你预想的高。消融 A8 会专门研究这一点。

**③ `n_agent=1` 时 GRPO 退化成没有 baseline 的 REINFORCE。**
`core_algos.py:143` 在组里只有一条样本时把均值设为 0、标准差设为 1，优势就等于原始奖励。这正好可以拿来做 baseline 作用的对照实验（A3）。

**④ `train_grpo.sh` 有个变量名 bug。**
脚本 export 的是 `DATA_DIR`，命令里用的却是 `$TRAIN_DATA_DIR` / `$TEST_DATA_DIR`，直接跑会找不到数据。建议从 `scripts/nq_hotpotqa/v0.2/train_grpo.sh` 复制一份作为基础模板，它是论文用的版本，用的是 `nq_hotpotqa_train` 数据。

## 1. 两种硬件配置

### 配置 A：8×A800（完整版）

| 项 | 设置 |
| --- | --- |
| 检索 | E5 Flat + `--faiss_gpu`，fp16 分片到 8 张卡（索引约 30GB，每卡约 4GB） |
| 训练 | `CUDA_VISIBLE_DEVICES=0-7`，`n_gpus_per_node=8` |
| 3B 超参 | 与 v0.2 脚本一致：`train_batch_size=512`、`n_agent=5`、`ppo_mini_batch_size=256`、`ppo_micro_batch_size=64`、`total_training_steps=1005`、`lr_warmup_steps_ratio=0.285` |
| vLLM | `gpu_memory_utilization=0.6`（给检索索引留出空间） |
| 用途 | 复现论文数值（R1、R2）、上 7B（S1）、关键结论的最终确认 |

**8 卡的另一种用法：拆成两个 4 卡槽位，同时跑一组 A/B 对照。** 检索服务只起一个（GPU 分片在 8 张卡上），两个训练分别用 `CUDA_VISIBLE_DEVICES=0,1,2,3` 和 `4,5,6,7`，都指向同一个 `retriever.url`。这样基线和变体在同一时间、同一检索负载下跑，比前后分开跑更公平。

> 两个训练都会调用 `ray.init()`。第一次这样跑时先确认两个 Ray 集群不冲突；如果冲突，就分别用 `ray start --head --port=<不同端口> --num-gpus=4` 起两个集群，再通过 `RAY_ADDRESS` 让各自的训练进程连上去。

### 配置 B：4×A800（快速版）

| 项 | 设置 |
| --- | --- |
| 检索（首选） | E5 Flat + `--faiss_gpu`，**检索进程也要设 `CUDA_VISIBLE_DEVICES=0-3`**，否则 faiss 会占用所有可见的卡（包括别人在用的）。每卡约 8GB，vLLM 降到 `gpu_memory_utilization=0.5` |
| 检索（备选） | E5 HNSW64 跑在 CPU 上（`example/retriever/retrieval_launch_ann.sh`），不占显存，但要大内存、检索更慢，召回略降 |
| 训练 | `n_gpus_per_node=4` |
| 3B 超参 | `train_batch_size=256`、`n_agent=5`、`ppo_mini_batch_size=128`（保持每步 10 次更新，off-policy 程度与配置 A 一致）、`ppo_micro_batch_size=32`、`log_prob_micro_batch_size=64`（这两个保证每卡负载与配置 A 相同）、`total_training_steps=300`、`lr_warmup_steps_ratio=0.1` |
| 用途 | 冒烟测试、代码阅读和调试、绝大多数消融实验 |

**配置 B 的结果只和配置 B 的基线（B-base）比**。batch 变小、步数变短，绝对 EM 会低于论文，这不影响判断消融方向。

### 选择配置的规则

- 要和论文数字对比，或者是 7B：只用 A。
- 消融实验：默认用 B；有 8 卡时用 A 的两个槽位并行跑两个 B 实验。
- 每卡负载保持一致（每卡 micro batch 相同），这样 4 卡 OOM 了 8 卡也会 OOM，反之亦然，排查时少一个变量。
- 第一次跑完 R1 后，把每步实际耗时填进第 6 节，再排后面的时间。下面的时间估计都只是数量级。

## 2. 第 0 组：跑通与观察（不追求分数）

| # | 实验 | 配置 | 做法 | 产出 |
| --- | --- | --- | --- | --- |
| X0 | 冒烟测试 | B（甚至 1 卡） | Qwen2.5-0.5B，`train_batch_size=16`、`n_agent=2`、`total_training_steps=5`，wandb 开着 | 链路跑通；wandb 指标名核对进 `learning_roadmap.md` 的指标表 |
| X1 | 轨迹解剖 | B | 在 `generation.py` 的 `_compose_final_output` 处加打印：解码一条轨迹，把 `info_mask=0` 的 token 标出来；在 `dp_actor.py` 打印每个 mini batch 的 shape | 一份带标注的轨迹样例，放 `notes/trajectory_anatomy.md` |
| X2 | 无训练基线（E0） | B | 未训练的 3B-Base / 3B-Instruct 加 `+trainer.val_only=true`，参考 `scripts/nq_hotpotqa/evaluate.sh`，`max_turns=4` | 7 个数据集的 EM 起点，后面所有提升都相对它算 |
| X3 | 耗时剖析 | A 和 B 各一次 | 跑 20 步 3B GRPO，记录 wandb 里 `timing_s/*`（gen、old_log_prob、ref、adv、update_actor）和检索服务的单次延迟 | 一张"每步时间去哪了"的表；看 4 卡是不是刚好比 8 卡慢一倍左右，如果远超一倍说明瓶颈在检索 |

**X1 是最值得花时间的一个**。agentic RL 和普通 RL 的区别全在这条轨迹里：哪些 token 是模型生成的、哪些是环境拼进来的、loss 落在哪里。

## 3. 第 1 组：复现（配置 A）

| # | 实验 | 改动 | 看什么 | 通过标准 |
| --- | --- | --- | --- | --- |
| R1 | ★ GRPO，Qwen2.5-3B-Base | v0.2 `train_grpo.sh`，`BASE_MODEL=Qwen/Qwen2.5-3B` | 奖励曲线、`env/number_of_valid_search`、响应长度、熵 | NQ / HotpotQA EM 与论文差距在几个点以内；差距大时先查检索器和数据版本 |
| R2 | ★ PPO，同一模型 | v0.2 `train_ppo.sh` | 加看 `critic/vf_explained_var`、`critic/vf_loss` | 能说清 PPO 与 GRPO 在收敛速度、稳定性、最终分数上的区别，以及 critic 前多少步是噪声 |

R1 跑的同时用 4 卡（如果有）跑 **B-base**：同样的模型、配置 B 的超参。B-base 是第 2、3 组所有实验的对照组，**必须先有它**。建议 B-base 跑两个 seed，量一下同配置下 run-to-run 的方差有多大。比这个方差小的差异都不算结论。

## 4. 第 2 组：机制消融（默认配置 B）

每个实验只列出相对 B-base 的改动。"预期"一栏在跑之前写好，跑完再写结论，对不上的地方往往是收获最大的。

| # | 主题 | 改动 | 主要看的指标 | 预期（跑前填写，下面是参考） |
| --- | --- | --- | --- | --- |
| A1 | ★ 检索 token mask | `actor.state_masking=false` | 奖励曲线、`pg_clipfrac`、KL、是否崩 | 更不稳，中后期可能崩；被 mask 的 token 在 π_θ 下概率很低，会把重要性比拉得很奇怪 |
| A2 | Base vs. Instruct | `BASE_MODEL=Qwen/Qwen2.5-3B-Instruct` | 前 50 步奖励、格式合法率、最终 EM | Instruct 起步快，最终差距不大 |
| A3 | ★ 组大小 / baseline | `n_agent` = 1 / 3 / 5 / 8，**同时设 `ppo_mini_batch_size = 32 × n_agent`**（即 32 / 96 / 160 / 256），让每步都是 8 次更新，这组内部自成对照，包括 `n_agent=5` | 奖励方差、每步耗时、组内全对或全错的比例 | `n_agent=1`（无 baseline 的 REINFORCE）明显更差、更抖；3→8 收益递减 |
| A4 | KL 系数 | `kl_loss_coef` = 0 / 0.001 / 0.01 | KL、熵、EM | 0 更快但后期可能漂移，0.01 学得慢 |
| A5 | 格式奖励 | 换 `main_ppo_format`，加 `reward_model.structure_format_score=0.2`、`final_format_score=0.1`（参考 v0.3 脚本），Base 和 Instruct 各一次 | 格式合法率、EM | 对 Base 帮助大，对 Instruct 帮助小 |
| A6 | 搜索轮数 | `max_turns` = 1 / 2 / 4，同时按脚本注释里的公式改 `max_prompt_length` | 分数据集 EM（NQ vs. HotpotQA / 2Wiki / MuSiQue）、平均搜索次数 | 多跳数据集收益更大；轮数越多每步越慢 |
| A7 | 检索质量 | BM25（`retrieval_launch_bm25.sh`）vs. E5；另做 `topk` = 1 / 3 / 5 | EM、搜索次数、`<information>` 长度 | 检索弱时模型会多搜几次，或者干脆不搜、靠参数知识猜 |
| A8 | ★ on-policy 程度 | `ppo_mini_batch_size` = 128（每步 10 次更新）/ 640（2 次）/ 1280（1 次，完全 on-policy） | `pg_clipfrac`、`ppo_kl`、样本效率（相同步数的 EM）、墙钟时间 | 更新越多，样本效率越高，但 clipfrac 和不稳定性也越高 |
| A9 | ★ 诱导 reward hacking | `main_ppo.py:27` 把 `compute_score_em` 换成 `compute_score_subem`（答案里包含标准答案就算对） | `<answer>` 平均长度、人工抽查 20 条 | 模型学会在答案里堆多个候选，训练奖励涨但严格 EM 不涨。**验证时仍然用严格 EM** |
| A10 | 温度 / 探索 | `rollout.temperature` = 0.7 / 1.0 / 1.2 | 熵、格式合法率、EM | 低温熵早早坍缩，高温格式乱 |

**优先级**：A1、A3、A8、A9 是理解 agentic RL 最关键的四个，先做；A2、A5、A6 次之；A4、A7、A10 有余力再做。

**不要在 3B 上把所有组合都跑一遍**。每个主题跑完写一段结论，再决定下一个。

## 5. 第 3 组：自己动手改算法（配置 B）

这组要改 `verl/trainer/ppo/core_algos.py`，每个改动都加一个配置开关，默认关闭，保证 B-base 不受影响。

| # | 改动 | 实现要点 | 验证方式 |
| --- | --- | --- | --- |
| M0 | 先加诊断指标 | 在 `compute_grpo_outcome_advantage` 里统计"组内奖励全相同"的组占比，记成 `adv/zero_adv_group_ratio` | 在 B-base 上看这个比例随训练怎么变：通常先下降，后期又上升（题目太简单或太难） |
| M1 | ★ Clip-higher（DAPO） | `compute_policy_loss` 的 `cliprange` 拆成 `clip_ratio_low=0.2`、`clip_ratio_high=0.28` | 熵曲线是否比 B-base 下降更慢，EM 是否持平或更好 |
| M2 | Dr. GRPO | 去掉除以 std 那一步（加 `norm_adv_by_std` 开关） | 对比响应长度和 EM；在全对或全错比例高的阶段差别最大 |
| M3 | 动态采样（简化版） | 在 `ray_trainer.py` 算完优势后丢掉零优势的组（不补采样，只是不参与更新） | 每步有效样本数、样本效率；结合 M0 的指标解读 |
| M4 | 超长惩罚 | 在 RewardManager 里对达到 `max_response_length` 的轨迹扣分 | 截断比例、响应长度 |

M1 和 M2 改动只有几行，适合用来确认你真的读懂了 loss；M3 要改训练循环，能帮你理解 DataProto 的流转。

## 6. 第 4 组：扩展

| # | 实验 | 配置 | 说明 |
| --- | --- | --- | --- |
| S1 | 上 7B | **仅 A** | 用第 2、3 组里最好的设置，Qwen2.5-7B-Base，和 R1 比收益与成本。7B 在 4 卡上理论上能跑（已经开了 offload），但会非常慢，不建议 |
| S2 | 新工具 | B | 在 `generation.py` 的 `execute_predictions` 里加一个 `<calculate>` 工具或换一个语料库，数据用需要计算或领域知识的 QA。重点是自己写环境反馈和奖励函数 |
| S3 | 换框架复现 | B | 用新版 veRL 的 multi-turn agent loop（或 SkyRL）把 B-base 重新跑一遍，对比实现差异和吞吐 |

## 7. 排期建议

按"4 卡是常态、偶尔有 8 卡"安排。有 8 卡的时候优先跑**只有 8 卡才能跑的东西**，其次是两个 4 卡槽位并行跑消融。

| 顺序 | 4 卡时 | 8 卡时 |
| --- | --- | --- |
| 1 | X0、X1、X2 | X3（A 部分） |
| 2 | B-base（seed 1） | R1 GRPO（完整） |
| 3 | B-base（seed 2）、M0 | R2 PPO（完整） |
| 4 | A1 → A3 → A8 → A9 | 两个槽位：A3 各组并行 / A8 各组并行 |
| 5 | M1、M2 | 两个槽位：A2 + A5 |
| 6 | A6、A7、M3 | S1 7B |
| 7 | S2 / S3 | 最好的配置用 A 跑一次完整确认 |

R1 跑完后，把实测时间填进这里：

| 实验 | 配置 | 每步耗时 | 总耗时 | 备注 |
| --- | --- | --- | --- | --- |
| R1 | A | | | |
| B-base | B | | | |

## 8. 每个实验的固定流程

1. 新建 `experiments/<编号>-<名字>.sh`，从 B-base（或 R1）脚本复制，只改动这一个实验涉及的参数。`EXPERIMENT_NAME` 带上编号和配置，例如 `B-A1-nomask-qwen3b`。
2. 在 `docs/experiment_log.md` 先写一行：编号、配置（A/B）、改了什么、**预期**。
3. 跑。每 50 步人工看 5–10 条轨迹（wandb 里打印的样例或日志里的 `Solution string`）。
4. 跑完填：git commit、wandb 链接、7 个数据集 EM、关键曲线截图或描述、**结论（与预期是否一致，为什么）**。
5. 单个 seed 下的差距小于 B-base 两个 seed 之间的差距时，结论写"不显著"，需要时补 seed。

## 9. 检查点：做完后应该能回答的问题

- [ ] 一条轨迹里哪些 token 进 attention、哪些进 loss？关掉 mask 后训练为什么会崩？（X1、A1）
- [ ] 默认 GRPO 配置下每步更新几次？这对 `pg_clipfrac` 有什么影响？（A8）
- [ ] 组均值 baseline 到底减少了多少方差？`n_agent=1` 时发生了什么？（A3）
- [ ] 模型在什么条件下会学会不搜索？（A7、A4）
- [ ] 一个宽松的奖励函数是怎么被 hack 的，从哪个指标能最早发现？（A9）
- [ ] 一个训练步的时间分布在 rollout、检索、训练三块的比例是多少？4 卡 vs. 8 卡的瓶颈是否一样？（X3）
- [ ] 自己改的算法有没有效果，效果是否超过 seed 方差？（M1–M3）
