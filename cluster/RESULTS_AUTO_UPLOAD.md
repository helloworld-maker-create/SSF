# 自动上传实验结果

`results.py` 是独立的 CPU 观察/发布入口，不 import torch、不申请 GPU、不读模型权重，也不改变训练、FSDP、KV、State、随机流、预算或健康门。
当前版本读取 Phase G 原生 B64、每5 outer更新G一次/C每outer更新的日志结构。G/C计数来自真正 accepted-update 记录，不能把 trainer iterations 当 G 更新。

## 上传什么

- 各 rank 共有、截至最后一个完整 critic/outer 边界的原始 accepted-update JSONL.gz。
- CSV、SVG loss 曲线、计数、EMA 状态、loss统计、Reader/Writer/backbone梯度组统计、torch allocated/reserved峰值。
- 过滤后的科学配置、source manifest、小型 COMMITTED checkpoint 清单和既有 hash。
- 每 rank 最多16KiB日志尾部，用于保留错误；完整日志继续留在服务器。
- 显式提供 `--eval-root` 时，附带 `evaluation/run.py` 最新的 summary、per-video CSV 和报告。

**不上传** Wan/teacher/fake-score/optimizer权重、视频、训练prompt文件、原始RNG/tensor、环境变量、私钥或用户健康评分。
不会反复扫描大权重。没有视频/分数时如实记录，不把loss曲线当质量结果，不伪造完成或健康PASS。
默认单文件/解压后上限8MiB、每包32MiB；越界停止发布，原数据保留，不自动删减证据。

## GitHub 凭据只需在运行服务器配置一次

复用师兄**服务器上已有的 GitHub Git 写权限凭据**。默认用SSH：

```text
git@github.com:helloworld-maker-create/SSF.git
```

若服务器原本用已认证的HTTPS Git，可以在首次执行前指定 `--remote https://github.com/helloworld-maker-create/SSF.git`。
不要把token写进命令URL、脚本、日志或Git。
本会话的GitHub连接不会自动转移到师兄的服务器，脚本也不会提取它的凭据。
403、SSH拒绝、hook拒绝、并发冲突或网络不确定结果后，当前自动流程停止所有后续上传尝试；不切换凭据、不force push、不盲目重试。
修好授权/连接并核对远端状态后，再手动执行一次 `publish`。已有远端快照内容相同会识别为已发布。

## 现有正在训练的任务：独立观察，不重启训练

观察器必须在**能直接读到run目录/共享文件系统**的机器上运行。
不需要改动已部署的 `.ssf_deploy/`、训练配置、源码manifest或当前worker。不要为加入发布脚本而重建/重启正在跑的实验。

如果当前工作分支是 `wubin`，可只取main上的两个独立脚本，放进一个新目录：

```bash
git fetch https://github.com/helloworld-maker-create/SSF.git main
mkdir -p /home/wubin/ssf_reporters
SSF_REPORTER_DIR=$(mktemp -d /home/wubin/ssf_reporters/version_XXXXXX)
git show FETCH_HEAD:cluster/results.py > "$SSF_REPORTER_DIR/results.py"
git show FETCH_HEAD:cluster/result_snapshot.py > "$SSF_REPORTER_DIR/result_snapshot.py"

PYTHONNOUSERSITE=1 python -B "$SSF_REPORTER_DIR/results.py" watch \
  --run-root /home/wubin/ssf_runs/20261026_131447_9d70a298 \
  --arm baseline --world-size 4 --interval-seconds 1800
```

这个例子的run_id来自已经发布的快照，**不表示该任务当前仍在运行**。执行前替换/核对真实run目录和world size。
每30分钟发布一次；所有rank有worker_finished记录或存在完整训练结束标记时，发布最终观察后退出。一个rank提前退出不等于整个任务完成。
本脚本前台运行；长期观察应由师兄已有的Slurm/batch会话管理。本次代码交付不会自动启动observer、controller、GPU任务或cron。

## 以后提交实验：一条命令包装现有launcher

先固定run_id，让 `--run-root` 与原launcher的输出一致。下面的尖括号需要替换，不改变原launcher的科学合同或GPU检查。

```bash
PYTHONNOUSERSITE=1 python -B cluster/results.py run \
  --run-root /shared/ssf_runs/<固定run_id> \
  --arm baseline --world-size 4 --interval-seconds 1800 \
  -- ./launch.sh <原先批准的launcher参数及相同run_id>
```

`--` 后原样执行原命令。GPU分配、资源预检、24h限制、checkpoint保存/恢复仍归原launcher或调度器；publisher不会替它们做决策。
定时和命令结束时自动保存/发布。不启动额外训练臂，不自动开始SSF或人工健康review后的后续阶段。
上传失败时训练子进程继续运行，之后仍保留本地快照但不再自动上传。不会kill、signal或重启任何训练进程。

命令退出码非0原样传递；原命令成功但收集/上传失败时返回3，请区分“实验失败”和“结果尚未发布”。
Ctrl-C会记录publisher中断，它不主动kill任务；训练/调度器仍须按原有生命周期管理，不应把publisher退出当GPU资源已释放。

## 自动评测之后也可以上传

对已有训练结果评分时，可用同样的 `run -- ...` 包装已有 `evaluation/run.py run ...` 命令，并附上 `--eval-root /shared/eval_output`。
评分脚本仍负责显式GPU选择与预检；publisher仅读取其结果，不额外评分或生成。
或在batch脚本评分结束后执行一次：

```bash
PYTHONNOUSERSITE=1 python -B cluster/results.py publish \
  --run-root /shared/ssf_runs/<固定run_id> --arm baseline --world-size 4 \
  --eval-root /shared/eval_output
```

## 只检查本地打包，不联网

```bash
PYTHONNOUSERSITE=1 python -B cluster/results.py snapshot \
  --run-root /shared/ssf_runs/<固定run_id> --arm baseline --world-size 4
```

`snapshot` 不执行任何Git/网络命令。单次上传用 `publish`；持续观察用 `watch`；包装一个已有命令用 `run`。
`--world-size`必须与实际训练一致；缺rank、跨rank不一致、非finite、计数或B64合同不符都不会被悄悄过滤成正常结果。

## 结果去哪里

默认分支：`results/<run_id>-<arm>`，例如 `results/20261026_131447_9d70a298-baseline`。
仅允许这个受约束的结果分支；不push到 `main`、`wubin` 或其他代码分支。

```text
GitHub result branch/
  RESULT_BRANCH.json
  latest.json
  README.md                          最新快照入口
  experiments/<run_id>/snapshots/<UTC+随机后缀>/
    README.md
    summary.json
    updates.csv
    loss.svg
    raw/updates_rank*.jsonl.gz
    logs/rank*_tail.log
    training_config.json
    checkpoint_manifest.json
    provenance/
    evaluation/                      如果已显式提供现有评测输出
    manifest.json                    文件字节数/SHA256及原日志前缀来源
```

旧快照不覆盖、不删除。Git只做普通fast-forward push；发生并发更新则拒绝并保留本地包，不自动变基、reset或force。
Git操作在新建的独立transport仓库中完成，不碰训练源码checkout、index、remote或脏改动，也不禁用Git push hooks。
根目录README/latest指针随新提交更新，历史版本和旧快照都在Git中保留。

本地默认状态目录是 `<run_root的父目录>/.ssf_result_uploads/<run_id>/<arm>/`，可用 `--state-dir` 指定持久目录。
里面保留每个package、transport Git对象和events.jsonl。它必须在训练run目录外；当前包失败也不会自动清理。
退出、time limit、OOM、权限失败都应结合snapshot状态、原日志和恢复标记判断；不要把worker的success=false直接称为模型“训崩”。

## 验证边界

已做CPU与本地bare Git测试，包括快照追加/幂等、并发拒绝、断行/缺rank/恢复计数、凭据检测、权重排除、上传失败不终止训练。
没有在本地启动真实训练、GPU评测或后台observer；没有代表师兄完成服务器Git认证。
新服务器首次远端push仍需师兄实际运行验证，不把本地测试冒充已打通其Git凭据。
