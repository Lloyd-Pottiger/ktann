# Cohere1M 当前基线、recall 归因与 cache 容量实验

日期：2026-09-26。源码基线：`3a62957c458a87b5b708af58ea1975c679e54ac2`。

后续实验：[rerank 150 / 125 / 100 对照](ann-rerank-2026-09-27.md)。125保留本批查询的逐查询recall，整体QPS收益较小；100引入额外候选损失。

本轮建立了可复用的固定索引，完成两轮性能曲线、6,000 次逐查询归因对照，并按用户允许合理内存投入的标准完成512 MiB/1 GiB cache 的 B/C/C/B 实验。生产算法和默认参数未修改；实验补丁保存在独立 worktree。

## 主要结果

- 完整构建 **1037.74 秒**。前台 routing 是已记录阶段中的主要成本；尚未实施 import 优化。
- 六个 beam 的 recall 损失全部发生在遍历阶段。候选筛选和精排没有额外丢失真邻居；本次查询集上提高精排上限无法补回缺失邻居。
- cache 上限从512 MiB增至1 GiB，在相同 recall@100=0.92889 下，beam384 的平均运行级 QPS **+32.8%**，CPU/query **-26.8%**，平均运行级 p95 **-37.1%**。实际 cache 计费约768.5 MiB，增加约256.6 MiB。

这里的性能结论限于本机、RocksDB、固定 Cohere1M、k=100、4并发、预热后查询。它是明确的内存换性能结果，不是建树质量改善，也不是算法在相同内存下的提升。

## 条件与证据控制

- Cohere1M，768维 cosine，1,000个不同 held-out queries，k=100，8个执行器 worker，4个并发搜索。
- 原生持续 Import Session：batch50、in-flight ceiling4、backlog watermark2、2个维护 worker、write beam8、Split Threshold128；与旧 bridge 的逐批 session/串行输入不同，不直接比较 import 提速。
- search budgets：Tree Key1、partitions1024、Leaf Entries65536、精排上限150；一个 Tree Key。每个性能点预热1,000次、测量2,000次。重复查询不增加独立 recall 样本。
- 两次正式六点曲线使用正常编译版；第二轮按固定随机种子打乱 beam 顺序。每次都校验 dataset identity、完整索引一致性，以及运行前后的逻辑 KV SHA256。
- 归因版独立编译，串行执行正常控制查询和带 collector 的查询。核对 IDs、距离位模式、usage、exhaustion 和 overlap truncation；诊断耗时不用于性能结论。
- 最终逻辑内容 hash：`2faed35a66813167e55d0761152a619afa7250a3f1a17d4c2eb6ff04f3e32655`。

构建后的首次搜索只作准备，不进入正式搜索基线。reuse 报告携带的 construction 数值属于原始构建，不是又执行了一次 import。[协议](experiments/cohere1m-2026-09/PROTOCOL.md)、[原始汇总](experiments/cohere1m-2026-09/results.json)。

## 完整 import

| 指标 | 结果 |
| --- | ---: |
| import | 1018.27 s |
| 后续维护收敛＋完整校验 | 19.47 s |
| complete construction | 1037.74 s |
| construction CPU | 1408.96 s |
| 进程峰值 RSS，包含先前 dataset loading | 4.32 GiB |
| import 逻辑读取量，前台和维护合计 | 716.63 GB |
| import 逻辑 mutation bytes | 5.36 GB |

最终验证1,000,000条记录、11,014个叶分区、123个level-2分区和1个root；所有分区Ready，无 actionable/transitional partition。完整构建累计搬迁1,436,954条 entry。后续 drain 约0.107秒，完整 topology verification 约19.357秒；本轮没有大规模未收敛维护债务。

20,578次前台尝试（20,000成功、578 retryable abort）的平均阶段耗时：routing33.07ms、prefetch6.84ms、apply0.48ms；成功 commit wait0.756ms。维护有56,039次成功尝试和1,671次 retryable abort。import 控制器182次升至2、182次降至1，没有达到配置上限4。

**推论与边界：** 应优先将写路由成本拆成重复读取、解码和距离计算，再测现有内部分区缓存是否可以复用。上述716GB是逻辑 backend 返回字节，不是 SSD 读取量，也尚未按 routing/maintenance 细分。admission 等待与有效工作重叠，不能直接把等待总和当成可节省时间。控制器并行度低值得实验，但本轮没有证明放开并发会改善完整构建或树质量。

## 同树搜索基线，512 MiB cache

下表范围来自两次独立运行，不是置信区间。单个 beam 每轮2,000次测量，包含同一批1,000个查询的重复。

| Beam | Recall@100 | QPS范围 | p95 ms范围 | p99 ms范围 | 平均 Leaf Entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.73253 | 194.87–203.53 | 24.30–25.50 | 26.83–27.91 | 5,925 |
| 96 | 0.78763 | 149.98–154.30 | 31.90–33.06 | 35.86–36.45 | 8,884 |
| 128 | 0.82316 | 122.19–132.11 | 37.51–41.99 | 42.43–53.65 | 11,845 |
| 192 | 0.86800 | 92.63–97.58 | 51.03–56.19 | 58.31–96.05 | 17,769 |
| 256 | 0.89617 | 76.46–77.27 | 64.66–66.01 | 73.02–75.81 | 23,697 |
| 384 | 0.92889 | 52.57–56.43 | 88.77–99.83 | 100.49–134.82 | 35,543 |

在这六个测试点中，只有384超过0.90。达到0.90的最小 beam 仍处于256和384之间，未测288/320，不能称384是全局最小值。两次 recall 完全相同，但 QPS 有数个百分点波动；小幅优化需要交错对照和 CPU/工作量证据。

## Recall 损失归因

每个 beam 的分母为100,000个 reference hits（1,000queries×100），逐查询分阶段损失之和等于 recall 缺口。

| Beam | 未进入已访问叶条目的 true hits | 局部/全局候选筛选损失 | 精排输出损失 |
| --- | ---: | ---: | ---: |
| 64 | 26,747 | 0 | 0 |
| 96 | 21,237 | 0 | 0 |
| 128 | 17,684 | 0 | 0 |
| 192 | 13,200 | 0 | 0 |
| 256 | 10,383 | 0 | 0 |
| 384 | 7,111 | 0 | 0 |

partitions 和 Leaf Entries 的硬预算均未耗尽。每次搜索都精排150个候选，且有 rerank exhaustion，但它并未导致本批 reference hits 丢失。不能只看 exhaustion 标志就推断精排候选不足。

**更具体的质量定位：** 当前只有123个level-2分区。beam256/384在上一层的宽度分别为128/192，因此已经覆盖全部level-2分区，枚举了所有叶 centroid；这两个点的剩余损失发生在叶分区选择。这排除了“只加宽上层 beam 就能修复目标 operating point”的解释。下一步应测叶 centroid 对实际成员的代表性、assigned-leaf versus nearest-centroid regret，以及训练/实际 placement 差异。本轮尚未区分 centroid drift、放置历史和高维几何各自的贡献。

## Cache 容量：有机制证据的配置收益

使用同一 capacity executable，以 B/C/C/B 顺序比较512 MiB和1 GiB，每轮测beam64与384。所有 recall、逻辑预算使用/耗尽以及索引内容都与原基线一致。[协议](experiments/cohere1m-2026-09/CAPACITY-PROTOCOL.md)、[比较数据](experiments/cohere1m-2026-09/results.json)、[各轮汇总](experiments/cohere1m-2026-09/results.json)。

| Beam | Recall@100 | 512 MiB QPS | 1 GiB QPS | 平均QPS变化 | p95 ms：512 MiB → 1 GiB | CPU/query变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.73253 | 185.57–200.15 | 217.25–233.84 | +16.9% | 24.72–26.72 → 19.58–21.33 | -19.9% |
| 384 | 0.92889 | 57.85–57.96 | 76.72–77.08 | +32.8% | 86.94–87.48 → 54.74–55.04 | -26.8% |

百分比比较两次运行的算术平均；p95变化也是运行级p95均值之比，不是合并样本的p95。
容量收益使用本组两次512MiB控制运行计算，不与前面的六点曲线混算。

beam384 的p99从97.30–97.37ms降至57.03–57.16ms。两次candidate的QPS均高于两次control，两次candidate的CPU/query、p95、p99均低于两次control。

在beam384，叶 cache miss从17.35%降至0.00%；每查询逻辑读取从约4.85MB降至0.49MB，CPU/query从约64.85ms降至47.47ms。这是减少重复加载/解码的证据，不只是调高容量后看到了更好的单次QPS。

实际计费 cache 从约512.0MiB增至768.5MiB；配置上限增加512MiB，实际多用约256.6MiB。测量窗口结束后的 worker RSS 样本，控制组为3707.4–3895.1MiB、1GiB组为3496.2–3782.7MiB。这些样本包括 allocator 保留的 dataset 内存等，不能将全部 RSS 差额归因于 cache，也不是搜索峰值内存。

**判定：** 在上述 warm-cache 工作负载上，容量调整的收益已展示；内存投入有界且实际用于避免重复加载。没有测冷启动/禁用cache/FoundationDB，不作这些场景的收益承诺。保留可复现实验配置，未修改生产默认参数。解码表示优化仍是另一个未验证候选，不应把本次容量收益算到它上面。

## 下一步

1. search 以1GiB这份有效配置作新对照，再分别验证解码缓存和叶 Header 批量读取，避免混合容量与实现收益。
2. import 单独 profile 写路由的读取/解码/计算，评估 snapshot/epoch 验证下的内部 centroid body 复用；不要直接放松 admission。
3. 质量实验优先做叶 centroid/成员分布诊断，再决定 refit、placement 训练或邻近重分配。仍需新建树、完整构建指标和相同 recall 的最终比较。

## 验证与复现

核心测试224通过、1忽略；benchmark测试38通过；相关 Clippy 和格式检查通过。独立审阅未发现 reuse-based 测量/归因的阻塞问题。小规模首次诊断只因“双查询指标样本数”校验失败，已修正并在小规模及完整1M重复验证；失败材料也保留。

所有数据与脚本位于 `.benchmark-data/results/ann-baseline-2026-09-26/`，属于本地 ignored artifacts。包括 frozen executables、patch、lockfile、provenance、reports、logs、fixed database及receipt。Rust1.85与当前主目录相同的Cargo.lock已记录。实验分支为 `codex/ann-baseline-2026-09-26`。

这是一个构建历史、一个数据集、一个后端的实测。import与树质量问题尚未修复；本轮新增的是可复用证据，以及一项经对照验证的缓存配置收益。

仓库内提供[实验补丁、结果与重放说明](experiments/cohere1m-2026-09/README.md)；大型原始材料保留在本地。
