# 官方 / 发布代码 / 当前本地 / Phase G 提案差异

上游commit固定33593df3e81fa3ec10239271dd2c100facac6de1；9文件远端内容匹配在前次只读审计完成。
本次仅保存Git固定commit与当前文件的小文件hash，不重扫旧大权重/视频。
引用：https://arxiv.org/html/2506.08009v2 ；https://github.com/guandeh17/Self-Forcing/tree/33593df3e81fa3ec10239271dd2c100facac6de1

|项目|Paper|Released code|Local actual|Phase G proposal|
|---|---|---|---|---|
|batch|DMD64, perGPU1|total_batch_size64未实现DMD累积|实际B1|G/C各B64，显式累积|
|GPU|64张H100，非100/200张|8nodes×8|单H20 NO_SHARD|单机2/4/8 FULL_SHARD资格后锁定|
|预算|短程post-training|README600，Trainer本身无限循环|G目标早退出|600完整outer，120G/600C|
|AdamW|2e-6/4e-7,β0/.999,eps1e-8,WD.01|一致|一致|一致|
|EMA|decay.99|outer200初始化，真实G后更新|旧run关闭、raw|FP32完整覆盖，400/600更新40/80次|
|初始化|ODE/teacher14B/fake1.3B|对应路径|对应资产|新起点，不接旧75/80|
|推理|prompt扩写协议|推荐详细prompt及use_ema|旧MVP原始prompt/raw|主EMA、次raw；参考详细另列|
|window|最后chunk不看首chunk的局部条件|YAML未设，默认-1|Local18/sink0|共同Local18，不推断发布权重实际训练窗|
|KV图|文字限制前帧KV梯度|无显式history detach|native旁路确实存在|不自动加detach，不称global TBPTT1|
|context|算法clean/t0|no_grad且scheduler加噪|同发布路径|min实际sigma约.00498008，保持不修|
|checkpointing|—|DMD开启|已证缓存重算副作用|两臂OFF，不修复|
|KV读|—|直接读可变缓存|grad读clone快照|保留当前梯度边与读值|
|cache transport|—|普通容器|opaque handle跨嵌套FSDP|保留|
|offload|未有完整合同|FSDP实现|冻结模型参数offload+activation offload|训练参数offload关闭，activation单独保留并测量|
|attention|FA3描述|按可用实现选|H20名字不命中H100/Hopper检查，默认FA2|不升级/修检测，记录实用实现|

训练文本248221行，仍是已发布/既有vidprom_filtered_extended；MVP不改写原始官方文本。
训练物理KV容量21latent，仅Local18有界读；推理物理18latent FIFO。两者不可混称训练物理驱逐。
旧结果只作能力/历史参照，不把新旧变化归因单一batch、EMA、预算或工程因素。
以上适配与paper/release差异须随科学合同批准，不能冒称严格官方等价。
