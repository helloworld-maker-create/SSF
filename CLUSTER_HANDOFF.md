# SSF 集群交接

本分支包含已验证的 Self-Forcing + Local18 / Persistent State 模型代码，以及真正使用过的 Phase G B64训练、EMA、完整恢复、监控和资格脚本。

新增 [自动评测入口与冻结测评数据](evaluation/README.md)：48/96/944 prompt 清单、官方维度标签、本地 lossless 视频评分及 Baseline/SSF 配对报告；不需要上传生成模型权重。

先读 [cluster/README.md](cluster/README.md)。代码与小型manifest进入Git；权重、数据、视频、完整恢复快照不进入Git。

一键训练入口为 `./launch.sh`，首次用 `./launch.sh --init` 生成本地配置，填写资产与持久输出路径后启动。它读取 `gpuall` 的 32 节点清单，通过 SSH 预检、同步代码、执行真实 DMD 资格验证，再启动 baseline；详见 [cluster/README.md](cluster/README.md)。

原两卡 H20 的 PASS 仍是历史证据，新拓扑必须在目标节点实测。`cluster/original_run/` 保持归档，不直接运行其中的旧 launcher。B4 提案没有实施。
