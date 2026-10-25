# 累积与EMA新增路径资格（待执行批准）

仅覆盖新增基础设施，不重做B–F，不运行routing/barrier/wrong-State/SVD矩阵。
资格预算48 GPU-hour是提案上限，不是批准；正式模型永远从批准ODE重新开始，不采用资格更新权重。

1. 小模型：64个已知样本，手工均值对分布式累积，检验G/C双方、参数固定、loss/A、无重复world除法及每更新一次clip/step/EMA。
2. 真实有界样例：选定拓扑下原生DMD固定权重/tape，流式手工FP32平均对累积路径，覆盖64个全局样本，不保存巨型全量梯度日志。
3. 所有rank参与相应根generator/fake_score的FSDP.clip_grad_norm_；禁止局部分片普通clip。核验最终同步含A-1次所有梯度。
4. SharedExit非恒定fixture[2,0,3,1,0,3,1]：same_step_across_blocks=True时所有7chunk与全部micro均使用index2。per-example noise/score timestep不改共享。
5. rank初始化、正确FSDP梯度同步、cache caller ownership与text K/V复用，只测新累积/拓扑下受影响路径。
6. EMA事件/名称覆盖825或836个参数、FP32、step200初始化、200 raw/EMA对应值相等、400/600刷新40/80次。
7. 临时资格G/C更新建立非空moments，完整保存/恢复，验证参数、moments、RNG、data/counters和配置；不进入科学训练臂。
8. 实测吞吐、host RAM、VRAM、初始化与保存开销；不复制八套旧单卡栈，选定最终拓扑前复核预算。

小模型FP32相对误差1e-5；真实梯度误差不超过max(参考范数1%,5×native-repeat绝对差)，同时必须通过同payload平均/更新控制。
该repeat尺度是工程参考，不是未来噪声置信界；独立backward不要求bit-exact，但状态/优化器/RNG/数据映射必须正确。
安装Torch2.8限制：参数CPU offload下不能在no_sync外直接累积；no_sync保留完整梯度且其间不CPU offload。generator/fake参数offload关闭，activation transport独立。
RNG必须重放真实样本，不能仅重设seed后假定相同；末次sync不能只保留最后micro。
新资格具体实现仍不存在。本文件不执行forward/backward，也不授予执行权。
