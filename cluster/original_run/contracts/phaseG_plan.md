# Phase G 最终合并合同 v1.2 — 待批准

状态：PLAN_FROZEN_PENDING_APPROVAL。此状态表示文件可供审批，不表示科学合同或执行获批。
本版本合并原Plan和定向修订；冲突时最新修订优先。原稿来源见history/。
本目录只含规划、状态、来源和CPU文档校验工具；不是训练worktree，也没有GPU/controller入口。

## 目标与不变量

先建立可用的参考对齐SF-Local18 Baseline，再只加入既有Persistent State完成一个matched MVP。
一个训练随机实现：seed2027、sampler0；两正式训练臂，Baseline先行、SSF有条件进入。
B64、micro1、600完整outer、120G/600C；EMA0.99，outer200初始化，400/600累计40/80次刷新。
State8tokens、1536维、Reader在block15后、Writer捕获block20；保持单转移detach及native历史KV。
保留真实DMD、原teacher/fake初始化、LR/优化器、scheduler/VAE、Local18/sink0，checkpointing OFF。
不得导入F routing，不能仅凭论文文字添加历史KV detach；不得新增seed、架构、loss、batch/EMA消融、30s、full944、wrong-State、SVD或大型归因。
旧R1/R2及F负结果完整保留，不宣布必然训崩；新旧差异不唯一归因B64、EMA或某项工程变化。

## 授权和顺序

科学合同批准、精简存储批准、阶段A批准、阶段B批准、用户评分/阶段C批准分别记录，互不替代。
approval_status_at_freeze.json全部PENDING；真实后续用户事件须绑定本bundle及相应合同SHA。
附件的<APPROVED_G_PLAN_PATH>尚是占位符；此候选Plan未被该占位符自动批准。
只获准整理、落盘规划及只读资源检查。48/1000/128 GPU-hour均为计划上限，不是当前授权。

阶段A（参考/新增资格）：48资格+8生成评分GPU-hour，136正常新视频。
G1更新吞吐/显存/host RAM/保存开销和完整成本后，需真实新批准才进入B。
阶段B（Baseline）：500训练+60生成评分GPU-hour，256正常新视频。
Baseline600完成，导出盲评包/空白表，封存AI与映射，置AWAITING_USER_HEALTH_REVIEW，释放GPU并退出。
阶段C需用户冻结评分、固定健康门PASS、明确新批准及资源复检；500训练+60生成评分GPU-hour，272新视频。
不启动后台等待GPU或自动越过人工门的controller。

## 参考与本地差异

官方commit33593df3e81fa3ec10239271dd2c100facac6de1。四列差异见official_local_gap_matrix.md。
发布DMD文件只含generator_ema：它是能力阳性参照，不是同预算竞争者，不取代ODE训练初始化。
保留8条已有详细prompt检查；再检查Local18及健康原始8prompt×2seed发布EMA。
ODE对应健康视频从ODE48×2复用；详细参考不混入原始MVP评分。
参照同任务失败标适用性问题，不剔除/换写prompt、不改健康阈值、不自动放行SSF。
H20/FA2、单机FSDP、Local18、checkpointing OFF、读快照/handle/offload是明确工程口径，不冒称官方完全等价。

## 累积、随机性、EMA与恢复

G/C各自64样本，A=64/world_size；一次更新内参数固定，loss/A反向，无第二次world_size除法。
前A-1次对应根FSDP.no_sync，最终同步包含所有先前micro梯度。所有rank调用对应根clip_grad_norm_，只clip/step一次。
禁止普通局部分片clip。micro之间不得推进outer、critic、optimizer或EMA，每个rollout独立reset KV/State。
同一逻辑G/C共享rollout计划：same_step_across_blocks=True时所有chunk都用exit_flags[0]；不是分别取列表项。
样本级noise/score timestep仍样本级。配对外生tape，不要求已学习预测和critic相同。
FULL_SHARD与参数offload/activation offload分开；具体字段及新资格只见training_contract.json与qualification文档。
真实G step后刷新EMA一次；初始化不是刷新，FP32覆盖全部Reader/Writer/token参数，不从旧稀疏权重伪造历史EMA。

精简例行完整快照只outer200/400/600；取消的是尚未产生的100/300/500，不删除历史。
保存raw、EMA/未初始化状态、critic、双optimizer/moments、所有rank RNG、sampler/cursor、next phase和计数、scheduler/配置/来源。
完整outer边界保存，accumulation_index=0；逐组件读回和SHA完成后COMMITTED。中断不能把部分梯度快照冒充完整。
每臂预留一份保留的应急完整快照，再额外预留一次临时写入。更多应急须重新核算/批准，不能覆盖旧证据。
若无法到安全边界则用最后COMMITTED重算，最坏近200outer约40G/200C；首次200前按冻结初始化重放。
失败/重算GPU时间计入额度；不能raw-only恢复，不能降B64或延长预算。

## 资源、MVP与停止

原存储门856GiB仍有效。候选精简方案：估峰482GiB，新增cap500+余量256=756GiB，单独批准前不得采用。
只用资源复检后真正空闲H20，GPU UUID启动即预留、独立端口；两臂锁定同一已资格拓扑8/4/2。
不盲目复制八套旧单卡host工作集。G1超成本返回BLOCKED_BUDGET；资源等候每60秒、最多30分钟。
不kill他人任务、不改环境、不删迁压旧证据；配额UNKNOWN不能当无限。

原始48prompt与seed2026/2027固定，主体16、人体动作8，其余四类各6；完整成员和ID哈希见evaluation_manifest.json。
主要endpoint为600 EMA对EMA，raw对raw次级，200/400仅固定8prompt监控。800是模型生成硬上限，正常664=A136+B256+C272。
同原4步/VAE、832x480、81帧、16fps、21latent、Local18/sink0、lossless RGB，配对initial及全部renoise。
HumanAction使用已验证原文文件名合同；无成员不评分，无Official Total/补零Quality或Semantic。
Bypass最多16辅助视频，不是MVP必须获胜条件；view-only只解决兼容性，不进入评分或模型生成样本数，记录源SHA/观看版本/存储。
评分T=0必须进入每视频严重失败并集；合理静态不是失败。健康门及暂停接口见baseline_health_gate.json。
AI必须标AI，用户是唯一人工，冻结前不泄露AI或模型映射；没有用户评分和明确阶段C批准绝不启动SSF。
健康FAIL/INCONCLUSIVE停止。有效负面MVP也完成执行，不换seed/调参/加样本。统计及结论见decision_rules.md。
完成已授权范围后停止；整个G0..G4尚未执行，不将文档冻结当整个实验完成。
