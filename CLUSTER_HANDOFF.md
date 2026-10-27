# SSF 集群交接

本分支包含已验证的 Self-Forcing + Local18 / Persistent State 模型代码，以及真正使用过的 Phase G B64训练、EMA、完整恢复、监控和资格脚本。

新增 [自动评测入口与冻结测评数据](evaluation/README.md)：48/96/944 prompt 清单、官方维度标签、本地 lossless 视频评分及 Baseline/SSF 配对报告；不需要上传生成模型权重。

[自动上传实验结果](cluster/RESULTS_AUTO_UPLOAD.md)：独立CPU观察器，定时/结束后发布日志、loss曲线、计数和既有评测报告到 `results/<run_id>-<arm>`；不上传权重或视频，不改正在运行的实验。

先读 [cluster/README.md](cluster/README.md)。代码与小型manifest进入Git；权重、数据、视频、完整恢复快照不进入Git。

这是可审计代码交接包。新集群的资源/环境/路径和提交方式尚待填写及资格；不能把原两卡H20的PASS当作集群PASS，归档脚本也不是已经适配的集群launcher。B4提案没有实施。
