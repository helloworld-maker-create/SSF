# 集群一键启动

仓库根目录的 `launch.sh` 是新入口。它从 `gpuall` 的节点清单选择已准备好的空闲节点，通过 SSH 同步代码，从 ODE 初始化直接启动 baseline；真实原生 DMD 专项验证改为可选。`cluster/original_run/` 的 51 个原始脚本、合同与依赖文件保持归档，不读取或伪造原主机审批。

## 第一次使用

```bash
git clone https://github.com/helloworld-maker-create/SSF.git
cd SSF
./launch.sh --init
# 编辑 cluster/site.local.json：Python、权重/数据、持久输出目录
./launch.sh
```

以后在同一仓库运行 `./launch.sh` 即创建一个新的 baseline 实验。初始化配置不会覆盖已有文件；本地配置、运行回执、权重、视频和完整 checkpoint 都不会进入 Git。

本机已核实为通过 SSH 管理的 32 节点池，每台 8 张 H100 80GB、约 2TB 内存。默认读取 `~/.config/gpuall/hosts`，可用 `gpuall --list` 更新清单。`--init` 会发现本机已有的 SSH identity、Python 和常用资产路径；发现路径不代表已通过字节或运行验证。通用模板是 `site.example.json`，其中的 `/REPLACE/` 必须替换。

默认先选一台节点的 4 张 GPU，每 rank 累积 16 次，G/C 仍为 B64。原因是原 activation offload 工作集按约 300GiB/rank 加 128GiB 主机余量准入，8 rank 同驻约 2TB 节点的资源假设尚未通过实测。`gpus_per_node`、`nodes` 均可配置；不自动减小有效 batch 或改配方。节点空闲状态每次启动重新查询，不会等待或终止他人任务。

## 配置与多节点

- `python`：目标节点上现成的训练 Python。参考 `environment_observed.json`，不使用历史 `requirements.txt` 自动升级环境。启动器会报告缺少的包，各节点版本必须一致，实际兼容性在启动时由运行结果反馈。
- `assets`：完整 Wan1.3B（模型、T5、tokenizer、VAE）、14B 六个 shards/index/config、ODE 初始化和 VidProM prompt。所有选中节点须能按配置中的绝对路径读取；程序不会下载权重、改符号链接或向 Git 上传大文件。
- `output`：目标节点上的持久路径，默认至少需要 320GiB 可用空间；tmpfs/ramfs 不通过。当前 0008 的磁盘空间不足，须准备其他持久位置或使用已准备好的其他节点。
- `transport`：`ssh` 或 `local`。SSH 使用配置的用户/identity 或现有 ssh config；无需 Slurm/PBS。`local` 只支持当前 checkout、单节点。
- `hosts`：留空则从 `inventory` 选择；也可指定 IP/主机名列表。候选节点必须有相同 Python 包版本及 GPU 型号。
- `environment`：显式设置集群需要的 `NCCL_*` / `GLOO_*` 等网络参数。不会把其他项目的网卡/HCA 设置直接套过来。

多节点示例：`nodes=2, gpus_per_node=4` 得到 world8、每 rank 累积8次；`nodes=32, gpus_per_node=2` 得到 world64、每 rank 一次。world 必须是 2/4/8/16/32/64，且各节点满足内存准入。B64 配方不支持将 32×8 卡同时作为一个实验的 256 rank。

多节点的 **output 必须是真正共享的持久目录**，同一个绝对路径应指向同一份内容。控制器会让每个节点写入本次随机标记，并从 leader 读回；只写相同路径字符串不能通过。这是完整 checkpoint、所有 rank RNG/sampler 和视频记录一致性的要求。代码不要求共享：小型源码会通过 rsync 自动同步到 `repo/.ssf_deploy/<source hash>/`，不覆盖其他 worktree，不同步权重或本地配置。SSH 控制机和目标节点需要 rsync。

## 命令与输出

```bash
./launch.sh --config /absolute/site.json --dry-run       # 只展示计划，不联系节点
./launch.sh --check                                    # 只读资产大小/包版本/GPU/RAM/磁盘预检
./launch.sh --qualify-only --run-id baseline_h100_01     # 实测后退出，不开始正式训练
./launch.sh --with-qualification --run-id baseline_h100_01 # 复用可选验证结果
./launch.sh --run-id baseline_h100_01 --resume /persistent/ssf_runs/baseline_h100_01/training/baseline/checkpoints/emergency_000016.pt
```

默认从头启动；绝不自动使用旧 world2 的 `emergency_000016.pt`。恢复只接受这个新运行、同一 arm、相同代码/配置/GPU UUID/世界规模的完整 `.pt` 与 `COMMITTED.json`，校验 SHA、双 optimizer、RNG、sampler/cursor、EMA 和调度器。已有 checkpoint 时必须显式指定 `--resume`。不覆盖历史文件；中断文件不算完整恢复点。控制机上的 `cluster/run_receipts/<run-id>.json` 保存对应运行信息，换控制机时需单独带上该回执。

输出在 `output/<run-id>/`：

- `run.json`、`source_manifest/`：本次代码/站点/拓扑指纹；归档资格不冒充本次资格。
- `node_checks/`、`handshakes/`：目标节点的资产字节检查、包版本和共享目录检查。
- `qualification/`：真实 DMD B64 梯度均值与精确前向/RNG 回放、单次 clip/step、State/cache 检查、完整保存后扰动恢复、EMA 覆盖、真实视频/VAE/编码耗时。
- `logs/qualify_*_rank*.log`、`logs/train_*_rank*.log`：各 rank 的 stdout/stderr；`training/<arm>/attempt_*/rank*/` 是逐 micro/update 审计。
- `training/<arm>/checkpoints/`、`generation/`：完整 checkpoint、原生 lossless 监控视频与记录。

可选验证使用可丢弃的 State 模型，正式 baseline 在独立进程重新从原 ODE/critic 起点构建，资格步骤不算正式训练更新。预计耗时取各 rank 的最慢观测，包含训练、conditioning、checkpoint、EMA 和监控视频并加 20% 余量；外部评分不在此预测内。预测超过 `wall_hours`（默认24）即停在验证结果，不开始正式训练；预测不等于完成时间承诺。原训练500 GPUh、生成60 GPUh、资格48 GPUh上限仍保留。

默认启动只检查必要资源与资产大小，不跑梯度回放/保存恢复/视频耗时专项验证，也不顺序扫描全部大权重 SHA。报错后停止本次进程组，在终端打印各 rank 日志尾部，并将结构化错误保存至控制机 `cluster/run_receipts/<run-id>.error-*.json`。不自动重试失败训练。需要专项验证时才加 `--with-qualification` 或 `--qualify-only`。

## 实验配方与 SSF

G/C effective batch64、microbatch1、600完整 outer（120G/600C）、seed2027/sampler0、Local18/sink0、原生DMD、ODE初始G、1.3B fake score、14B teacher。Generator/fake-score checkpointing OFF，FSDP参数CPU offload OFF，saved-tensor activation offload ON。每 micro 原生 FP32 同步累积，每个逻辑 batch 只做一次根 FSDP clip/optimizer step。

EMA=.99，outer200初始化，400/600分别40/80次真正G后更新。State仍8 tokens、Reader15/Writer20、单转移detach。新拓扑使用 seed2027+global rank 的新随机流，不声称跨拓扑精确续训。多 rank 监控用不写入结果的补齐调用保持集合通信次数相同，不增加或重复冻结评测行。

默认只启动 baseline。600完成后退出并释放 GPU，评分仍需用户实际完成。原64条匿名健康包还包含 ODE/发布版 DMD 参考视频；这些参考视频没有随源码或训练资产清单提供，需另行准备后按原合同导出，不能用 baseline 视频重复填充64条。评分可使用仓库已有的 `evaluation/` 入口。训练完成不会自动产生健康 PASS 或启动 SSF。

SSF 需显式 `--arm ssf`，且 `ssf_review` 指向用户已有的真实冻结审阅记录，包含：

```json
{
  "origin": "actual user review",
  "status": "PASS",
  "ssf_authorized": true,
  "baseline_run": "/persistent/ssf_runs/baseline_h100_01",
  "scores_file": "/absolute/user_scores_frozen.json",
  "scores_sha256": "填写实际冻结评分文件的SHA256"
}
```

启动器只读取并校验这些材料，不生成评分或批准。正式 SSF 从匹配起点重新训练；专项验证同样由显式选项控制。

## 源码与验证边界

`original_run/` 原始51文件及原科学合同保持字节一致。模型源码适配包括 `utils/wan_wrapper.py` 的可选资产根路径，以及 `wan/modules/attention.py` 对 FlashAttention 3 的 Tensor/元组返回格式兼容；未设置资产根路径时仍用原默认目录。改动单列在 `qualified_source_manifest.json` 的 `portability_overrides`，不伪称95文件全部字节不变。DMD、State、KV、采样和训练配方保持原样。

新入口还未经目标集群真实训练验证；随代码提供 CPU 启动器测试和原累积回归，实际 GPU PASS 只能由该次运行的 `qualification/.../complete.json` 产生。
