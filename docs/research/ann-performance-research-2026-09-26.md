# KTANN import、树质量与 search 性能研究

日期：2026-09-26。源码基线：`3a62957c458a87b5b708af58ea1975c679e54ac2`。

后续已完成当前版本的1M实测、recall归因及cache容量对照，见[基线与实验结果](ann-baseline-2026-09-26.md)。以下保留最初研究阶段的证据与假设。

本轮完成 issue 最新记录、现有实现、历史原始报告和相关论文的交叉核查，并重新计算历史指标；没有修改生产实现，也没有运行新的性能 benchmark。下面明确区分已观测结果、源码事实和待验证假设。

## 结论与验收目标

优先解决两件事：降低达到目标 recall 所需的遍历工作量；减少同一候选集合下重复评分和 KV 往返。import 同时区分在线建树成本与调用方的输入/提交调度。现有证据不足以把全部问题归因于某一个训练参数。

用户已明确接受适度内存增长和仅在有 cache 场景成立的收益。主要目标是完整 import 更快、树质量更高、相同实际 recall 下 QPS 更高且 latency 更低，资源在合理范围内并被有效使用。因此：

- 以目标工作负载的 recall–QPS–latency 曲线为主；缓存收益可以独立成立。
- 记录 cache 容量、进程 RSS、冷填充成本和禁用 cache 的表现；不要求所有场景都提速，也不把小幅资源增长自动视为失败。
- 有界且计费准确的内存换计算是可接受方向。未计费的持有、无界增长、错误快照复用、破坏 exact membership 仍不可接受。
- import 计时包含提交、维护收敛与完整校验，不能只优化前台时间而掩盖更大的后续工作。
- 明显的尾延迟或资源退化需要说明影响和收益；这不等于用户要求每个指标都零退化。

## 1. 先统一比较对象

### 最新 VectorDBBench 记录是 recall@100

本地 `issue-128/cohere1m-beam384` 保存了用户描述的工作负载：Cohere1M、768 维 cosine、k=100、1,000 个查询、RocksDB、512 MiB Partition Cache、batch=50、load concurrency=1。并发测试每档仅 5 秒。

| 指标 | 历史记录值 |
| --- | ---: |
| leaf beam | 384 |
| recall@100 | 0.9293 |
| QPS，concurrency=4 | 45.6125 |
| concurrent p95，concurrency=4 | 119.0 ms |
| canonical load duration | 1,495.50 s |
| bridge 累计 native import 时间 | 1,058.12 s |
| bridge optimize 时间 | 0.111 s |
| 全部 1,266 次 search 的平均 visited partitions | 505 |
| 平均 visited Leaf Entries | 35,426.12 |
| 平均 exact rerank candidates | 150 |

这些是既有报告，不是当前 HEAD 的新基线。`invocation.json` 记录 revision `331aef4`、独立 binary SHA256；目录另有默认 beam 改为 384 的实验 patch，同时命令显式指定 384。不能仅凭 revision 字段把该二进制当成干净的 `331aef4` 或当前 HEAD。当前源码默认 beam 是 128。

另一份 beam=32 记录为 recall@100=0.6279、4 并发 104.4039 QPS，但它是另一棵树，search budgets 也不同。两者只能说明现有 operating points，不能当成严格的单变量比较，更不能证明 384 是达到 0.9 的最小 beam。应在同一棵树上补齐 64/96/128/192/256/384 的曲线。

最新树有 11,037 个叶分区、120 个 level-2 分区和一个 root。`505 = 1 + 120 + 384`，与当前 beam 逐层减半、384 在上一层允许 192 个分区的规则吻合：这个点已覆盖全部 level-2 分区，再筛选 384 个叶分区。**推论**：此点存在很大的叶质心评分与 Leaf Entry 评分工作量；但仅凭聚合数不能分配各阶段 CPU 时间。

这份 bridge 的 topology 是 Header snapshot，明确声明不是 full integrity verification。因此 0.111 秒 optimize 不能解释为完整校验只需这么久，也不能与 native harness 的 complete construction 指标混同。

来源：本地 beam384 canonical report（本地材料：`.benchmark-data/results/issue-128/cohere1m-beam384/canonical/KTANN/result_20260926_KTANN%20plus%20benchmark%20bridge_ktann.json`）、bridge report（本地材料：`.benchmark-data/results/issue-128/cohere1m-beam384/bridge.json`）、invocation（本地材料：`.benchmark-data/results/issue-128/cohere1m-beam384/invocation.json`）、beam32 report（本地材料：`.benchmark-data/results/issue-128/cohere1m-rocksdb/bridge.json`）。这些路径属于本地 ignored artifacts，不是仓库发布附件。

### 旧的 recall@10 结果不能直接解释 recall@100

2026-09-13 的诊断在一棵 settled Cohere1M 树上记录了：

| beam | recall@10 | 因 traversal 未访问而丢失的 true hits / 10,000 | 后续候选选择和精排丢失 |
| --- | ---: | ---: | ---: |
| 8 | 0.5511 | 4,489 | 0 |
| 32 | 0.7704 | 2,296 | 0 |
| 128 | 0.9007 | 993 | 0 |
| 144 | 0.9083 | 917 | 0 |
| 192 | 0.9274 | 726 | 0 |

该诊断对 collector 开/关的同树结果、距离位模式和预算做了对照；本轮重新核对了原始 report/log 的 SHA256 与分阶段损失之和。它支持“那棵树的 k=10 损失发生在遍历阶段”。它不证明新树 k=100 也如此。

当前精排规则为 `max(64, k + ceil(k/2))`，所以 k=10 时为 64，k=100 时为 150。最新 k=100 的每次 search 都使用 150 个精排候选并报告 rerank exhaustion，**但 exhaustion 不等于丢失真邻居**：旧 k=10 诊断也大量触发该标志，却没有因此丢失 true hits。需要阶段级 survival，而不是据此直接提高 cap。

来源：历史 attribution（本地材料：`.benchmark-data/results/issue-145-critical-path-2026-09-13/quality-full-attribution.json`）、[ADR 0011](../adr/0011-bounded-deterministic-approximate-search.md)、本轮重新汇总的 evidence（本地材料：`.benchmark-data/results/ann-research-2026-09-26/evidence.json`）。

## 2. import 与建树：有证据支持哪些方向

### 不再把 admission 参数调大直接等同于 import 提速

[#156 最新正文](https://github.com/Lloyd-Pottiger/ktann/issues/156) 使用 `f640875`、200k Cohere、k=10 的筛选工作负载。放宽 retry backlog 条件曾令前台 import 降至 14.30 秒，但 complete construction 从 79.08 秒增至 207.94 秒，移动记录约增 7 倍。另一种局部维护优先方案也未通过整体评估。这否定的是特定实现，不是否定所有调度优化。

原始实现将 new-submit watermark 从 2 改为 16 的一次试验，complete construction 约下降 9.9%，CPU 约上升 6.0%，mutation bytes 约上升 2.8%。按用户的新取舍，这值得复测；它不等价于放松 retry quiescence，也不是已证明可推广的默认值。

### 训练目标与实际放置目标不同

当前 split 对完整 source snapshot 训练，每轮按距离差选出恰好一半记录，返回两个质心；实际 drain 采用最近质心，外加达到最小占用的 viability reservation。后续插入也按路由选择叶分区。训练的 balanced mask 没有持久化。[源码审计](current-ann-path-audit.md) 给出位置和约束。

这构成可测假设：balanced assignment 与最终 placement 的差异，可能让实际分区失衡、质心代表性变差，进而造成重复 split 和更大的 search beam。不能仅由算法形式断定它是主要瓶颈。

第一步测量真实 split snapshot 的 label disagreement、nearest-placement 两侧数量、实际 drain 分布、split 后短期再次 split/merge 比例，以及每条记录的移动次数。然后保持种子和数据顺序一致，只比较现有训练与按实际 placement 优化的训练。内部节点与叶节点分开统计。修改公开参数或放弃 occupancy 约束都不是这个实验的前提。

### Immutable centroid 与构建历史

非 root centroid 不随成员变化更新是 [ADR 0015](../adr/0015-incremental-binary-kmeans-tree.md) 的既有决策。应测量存储 centroid 与当前成员中心的漂移、祖先路由丢失、不同导入历史的方差。离线 refit 可以估算质量上限，但在线刷新需要同时考虑父 Child Entry、epoch、原子性和并发维护；不能直接加一个后台“重算中心”任务。

已有旧版实验包含 spherical/normalized centroid、更多 Lloyd rounds、各层相同 beam。不能把这些重新包装成从未尝试的修复；需要新的归因支持，并检查其旧版、dirty revision、饱和 recall 等限制。

### Bridge 路径需要单独对齐

[bridge insert handler](../../benchmarks/src/bridge.rs#L452) 每个至多 50 条的请求在 state write lock 下创建 session、submit、finish。原生持续 Import Session benchmark 与之不同。最新记录还使用 load concurrency=1，因此单纯保留 session 不会凭空产生更多并行工作；必须区分会话开销、调用端串行供给、锁与 maintenance 的相互影响。

canonical load 1,495.50 秒与 native import 累计 1,058.12 秒之间有约 437.38 秒差额，但它混合了数据读取、客户端转换/编码、传输、调度和不同计时边界，不能全归为 JSON 开销。下一轮应拆分 native committed batch、source/encode、等待、maintenance progress，保持“成功响应已提交”的语义。

## 3. search：重新评估缓存收益，并降低 KV 往返

### 已经测到收益的 decoded-cache 候选值得重启

[#170 的 bounded-ID-block 实验](https://github.com/Lloyd-Pottiger/ktann/issues/170#issuecomment-5826855775) 在固定 SIFT1M、相同 recall 下，beam32 的 QPS +6.09%、CPU -5.42%；当时主要因为冷 Partition Cache 的短 burst p95 +5.89% 而撤回。它不同于更早因 cache pressure、对象持有等问题失败的展开方案。

按用户的新验收，这不应再被自动淘汰。优先在当前 HEAD 的 Cohere768d、k=100 目标点复测，明确缓存表示增加多少 bytes、能装下多少 body、命中率和稳态收益。旧实验的 Fashion784d 基本持平，不能把 SIFT128d 的 +6.09% 当成 Cohere 的预计收益。若适度提高 cache 容量能带来更好的固定-recall 曲线，可作为明确的内存换性能方案评估；实现变化和容量变化分别对照。

ownership 修复仍必须保留：候选 ID 不应意外持有整个已淘汰 partition arena，所有额外表示纳入 cache accounting。用户接受资源投入，并不使这类资源生命周期问题合理化。

### Leaf Header 批量读取是另一个直接候选

当前实现已批量读取 internal Headers，FDB `batch_get` 已并发处理点读；leaf Header 仍在遍历中逐个读取。cache hit 也需要 Header 的 snapshot/epoch 校验。对于平均 384 个叶分区的工作负载，减少这类依赖往返可能有价值。[源码审计](current-ann-path-audit.md) 列出位置。

先测 warm cached search 的 Header 请求批次、等待时间与 CPU 占比，再尝试只合并已经预算许可的有限数量读取。必须保持同一 snapshot、相同候选/预算/错误语义；不能提前读取并暴露未获预算许可分区的 corruption。收益需要体现在 QPS 和尾延迟，不能只以 RPC 数下降验收。

### 当前数据不支持先优化 Python/IPC search

最新实验全部 search 的 client round-trip 累计 94.8079 秒，native search 累计 94.1535 秒，差额约 0.69%。这些是跨查询的 duration sums，不是并发阶段的 wall time，也不是纯 socket 延迟。它们支持优先调查引擎内 search。import 的计时差额则不能沿用这一结论。

来源：client summary（本地材料：`.benchmark-data/results/issue-128/cohere1m-beam384/client-summary.json`）。

## 4. 具体实验顺序

1. **当前基线与 recall 归因。** 固定 HEAD、binary hash、数据/查询顺序、k=100、cache 容量、并发、预算；在同一 settled tree 上跑 beam 曲线。记录正常遍历覆盖的 true hits，离线对这些叶中的记录精确评分得到 traversal ceiling，再记录 local/global candidate survival 与最终 hits。instrumented 与正常路径核对 IDs、distance bits、预算和错误；诊断耗时不作为性能结果。
2. **可独立验证的 search 实验。** 分别复测 bounded decoded cache、已获预算叶 Header 批量读取。每次只变一个机制；固定树及查询，交错 B/C/C/B，使用稳定预热与长于旧 5 秒的测量，报告运行间分布。收益主要在 cache 场景即可成立，但明确内存和冷启动代价。
3. **建树实验。** 对真实 split 加仅用于实验的统计，先检测 placement mismatch 和 centroid drift；再各自做单变量 ablation。导入 watermark 2/16 是成本较低的独立对照，不能与训练变化混在一起。先小规模筛选，再 Cohere1M、多次独立重建；固定记录顺序仍不能消除异步构建方差。
4. **整体确认。** 按每棵树达到 recall@100 >=0.90 的真实 operating point 比较 QPS、p50/p95/p99，并同时报告 complete import、CPU、峰值 RSS、cache bytes、重试、写入/移动量。禁止把不同 k 的 recall、不同树的固定-beam QPS 或插值结果作为提升证明。再验证 SIFT 与 FoundationDB，区分客户端和服务端资源。

初始诊断的 recall 与后续实验选择使用同一 1,000-query 集会形成调参偏差；最终应保留独立查询验证，或明确只在该固定 benchmark 上成立。重复查询可降低计时噪声，不增加独立 recall 样本。

## 5. 架构候选与边界

[论文研究](ann-tree-quality-prior-art.md) 的主要启发是：SPANN 的效果不只来自 balanced clustering，还依赖边界复制和路由；直接复制记录违反 KTANN exact one-leaf membership。SPFresh 的邻近分区重分配更适合作为后续候选，但要计算移动量、事务边界和稳定性。DiskANN 提醒我们区分“搜索宽度”与“I/O 并行度”。

对于空索引 import，离线高质量聚类加自底向上 materialization 值得用作质量/构建成本的对照上限，但正式 bulk build 会改变现有“所有 import 走在线 mutation、所有 committed state 可搜索”的生命周期设计，需要先完成可审阅设计。它不是本轮默认引入的实现，也不应因当前没有稳定版本而忽略原子性和恢复契约。

## 可复核材料

- [实现审计与反证实验](current-ann-path-audit.md)。
- [相关论文与可迁移机制](ann-tree-quality-prior-art.md)。
- 历史证据汇总（本地材料：`.benchmark-data/results/ann-research-2026-09-26/evidence.json`），包含输入文件 SHA256；重新计算脚本（本地材料：`.benchmark-data/results/ann-research-2026-09-26/analyze.py`） 只读历史数据。
- 最新 issue 记录已从 GitHub 获取；本轮不更新 issue、不改变默认参数、不复用未知来源二进制作新性能声明。

本轮验证：历史 attribution report/log hash、阶段损失总和、原始 canonical 与 companion 指标、源码引用与文档 diff。研究文档变更无需 Cargo 检查。尚未验证的是当前 HEAD 的新 baseline、候选实现的收益与新树的 recall 归因，不能把研究完成称为性能问题已解决。
