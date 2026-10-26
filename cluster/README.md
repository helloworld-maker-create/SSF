# 集群交接说明

自动评测与数据见 [evaluation/README.md](../evaluation/README.md)：可直接对服务器本地 lossless 原片评分与生成配对报告。权重、原片和训练数据不通过 Git 传输；48/96/944 prompt 清单与官方 metadata 已包含。

## 已包含

- 仓库根目录：实际通过B1/B1.1/B2和真实Wan/DMD/FSDP验证的完整模型、KV handle、State及原生loss源代码；不是仅上游的干净代码。
- `qualified_source_manifest.json`：原95文件哈希；仅`.gitignore`为打包保护调整，模型/损失/调度代码保持字节一致。
- `original_run/scripts/`：真实Phase G训练、B64同步累积、EMA、完整恢复、监控、评分、资格和CPU测试代码。
- `original_run/contracts/`：原冻结科学合同及48prompt/664视频矩阵。历史PENDING字段与原主机审批分开；审批记录和人工评分没有移植或伪造。
- `original_run/vendor/`：原评分工具依赖及官方membership metadata。
- `assets_manifest.json`、`environment_observed.json`：独立传输与环境核对清单。
- `evidence/`：原主机G1通过记录和4G/16C完整恢复点的小型审计记录；均不代表集群资格。

## 科学合同保持B64

G/C均effective batch64，microbatch1，600完整outer=120G/600C，seed2027/sampler0，Local18/sink0，原生DMD、ODE初始G、原始1.3B fake score、14B teacher。Generator/fake-score checkpointing OFF；FSDP参数CPU offload OFF，saved-tensor activation offload ON。同步每micro的FP32累积已在原world2上资格，最后只调用相应根FSDP.clip_grad_norm_、optimizer.step各一次。不能直接使用历史`train.py`就认为total_batch_size=64已经生效。

EMA=.99，完成outer200初始化，400/600分别40/80次真正G后的EMA更新；State仍8 tokens、Reader在block15后、Writer capture20，直接State单转移detach，native KV梯度不变。无F routing，无B4改动。

Baseline先行；600完成后必须导出64完整匿名视频和空白用户评分表，释放资源，等待用户实际冻结评分及明确SSF批准。不得克隆原审批当作新的集群授权。

## 权重与数据单独传输

需要完整Wan1.3B目录（含T5、tokenizer、VAE）、Wan14B全部6shards及index/config、ODE和VidProM prompt文件；逐项字节数/哈希见`assets_manifest.json`。仅实际加载前进行全SHA校验，不每步重扫。禁止通过Git上传大权重/视频；不自动下载或修改符号链接。

可选完整恢复点为原主机`emergency_000016.pt`，31.727GiB，SHA256 `115729b0b8d753dd021b0013184c6ccff726ef4cfb9549e2820cbeefbe73e92b`，包含G4/C16、双optimizer FP32 moments、RNG、cursor1280，EMA尚未初始化。它不是600结果，也不是raw-generator-only模型。

原恢复点只有两rank RNG/sampler，**不能直接在8/64卡上声称精确续训**。新拓扑需独立资格和明确随机流方案；更大拓扑的匹配实验通常应从同一ODE/critic起点重新做两臂，并保留这次工程前缀，不可冒充无缝恢复。

## 当前可执行的安全检查

```bash
PYTHONNOUSERSITE=1 /path/to/python -B cluster/preflight.py
# 填写assets.example.json的路径后另存为未跟踪的assets.local.json：
PYTHONNOUSERSITE=1 /path/to/python -B cluster/preflight.py --assets cluster/assets.local.json
# 只有需要完整字节验证时才加 --verify-weight-sha256；会顺序读取所有大文件。
```

默认预检只用Python标准库，不import/实例化模型，不分配GPU、不训练、不下载。`--resources`只查询nvidia-smi和主机内存/磁盘。通过只代表静态代码/资产检查，绝不是DMD或集群运行PASS。

## 新集群启动前必须补齐

1. GPU型号、数量、单卡VRAM、单节点RAM、节点数、Slurm/PBS及真实资产目录。不能按更多卡线性承诺24h，activation offload的host工作集也需要实测。
2. 原运行代码里的本机路径、两卡/world2硬门、H20/120GiB检查、Python路径、历史审批和local manifest都必须做受审查的站点适配；**不要直接运行`original_run/scripts/phaseg_launch.py`**。它们按原样归档，用来防止漏带真实代码，不是通用集群启动器。
3. 只重验新增环境/拓扑/累积路径：小型真实DMD前反向、均值/裁剪/step、cache ownership、State梯度、EMA coverage、主机/显存和吞吐。禁止用mock损失替代DMD，禁止为速度悄悄减batch或改变schedule。
4. 实测完整每臂成本能满足24h才启动。额外GPU和空间需要集群实际分配，不能仅凭账号access推定可用。

## 容量与安全

当前原主机/data剩余约154GiB，完整两臂六个恢复点及视频按观测预计约273GiB（含32GiB余量），缺约119GiB。集群请单独分配持久输出空间；不要用内存盘或会自动清理的目录保存唯一完整checkpoint。禁止删除旧证据或杀他人进程。

历史requirements.txt并非新的通用安装器。`environment_observed.json`记录原主机实测版本（包括transformers4.45.2与历史requirements>=4.49的差异）；在集群先审查环境，不要盲目升级或安装可选UI/TensorRT依赖。

## 发布

建议推送到用户自己的私有仓库，再给师兄访问权限。上游guandeh17/Self-Forcing只是来源，不是本次发布目标。本分支没有添加用户私有remote，也没有推送到上游。
