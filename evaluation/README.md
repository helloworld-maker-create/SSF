# 自动评测：数据先行，权重留在运行服务器

本目录新增的是**对服务器上已生成的 lossless RGB 视频自动评分与配对汇总**的入口。
它不加载 Wan/T5/teacher，不训练、不生成视频，也不上传 checkpoint。
训练/推理继续使用服务器本地权重；生成原片之后调用本入口即可。

## 已上传的数据

| 文件 | 用途 | 原样保留的 prompt 数 | 默认 inference seeds |
|---|---|---:|---|
| `data/phaseG48.json` | 当前 Phase G MVP | 48 | 2026 / 2027 |
| `data/phaseCD96.json` | 既有 C+D/F 面板的复用 | 96 | 2026 / 2027 / 2028 |
| `data/official944.json` | 原官方唯一 prompt 池 | 944 | 原完整评测 seed 2026 |

没有新增、扩写或按实验胜负重选 prompt。原始官方 946 条 metadata（944 个不同文本，两个跨维度重复文本）继续复用
`../cluster/original_run/vendor/official_vbench_prompt_metadata.json`，包括 object/action 等 auxiliary 信息。
`data/provenance.json` 记录冻结来源和 SHA256。提供 944 清单不表示自动获准重跑完整 benchmark。

三个 JSON 还附有同名 `.txt`，文本及顺序逐条一致，方便服务器端读取；prompt ID、官方维度和 auxiliary 信息以 JSON/官方 metadata 为准。

默认 `config.example.json` 使用 Phase G 48×2、Baseline/SSF 两臂，共**192 个已有视频**，不是新生成任务。
维度覆盖为：subject 16、background 6、object class 6、multiple objects 6、human action 8、dynamic degree 16、motion smoothness 16、imaging quality 6、overall consistency 6。

## 先查看清单：无需 GPU、视频或权重

```bash
PYTHONNOUSERSITE=1 python -B evaluation/run.py plan --config evaluation/config.example.json
PYTHONNOUSERSITE=1 python -B evaluation/test_evaluation.py
```

`plan` 和 CPU 测试只用 Python 标准库，不 import torch/vbench，不访问 GPU，不下载。

## 配置服务器已有视频

把 example 复制到已有 `.gitignore` 保护的 `outputs/` 内，再填写本地路径：

```bash
mkdir -p outputs
cp evaluation/config.example.json outputs/evaluation.local.json
```

| 配置 | 内容 |
|---|---|
| `video_root` | lossless 原片及逐视频 generation records 的共同根目录 |
| `video_pattern` | 默认 `{arm}/{seed}/{prompt_id}.mp4`，可改成已有实际目录模式 |
| `record_pattern` | 默认 `{arm}/{seed}/{prompt_id}.json`，可改实际 generation-record 目录模式 |
| `output` | 新的持久评测输出目录；Human Action hardlink 要求与原片处于同一文件系统 |
| `metric_python` | 服务器上已有 VBench 环境 Python；可与生成环境分开 |
| `vbench_cache` | 已有 DINO/CLIP/RAFT/GRIT/UMT/ViCLIP 等评测资产目录 |
| `arms[].checkpoint_sha256` | 从训练/生成 manifest 填入真实 SHA256；不会打开权重文件 |
| `arms[].weight_kind` | 两臂均 raw 或均 EMA；不能混比 |
| `panel` | 相对本 evaluation 目录的数据路径，或自定义冻结清单的绝对路径 |
| `seeds` | 显式填写所选面板的 inference seeds，不自动增加 seed |

目录举例：

```text
/storage/lossless_videos/
  baseline/2026/p0169-2f9950b3d9a7.mp4
  baseline/2026/p0169-2f9950b3d9a7.json
  ssf/2026/p0169-2f9950b3d9a7.mp4
  ssf/2026/p0169-2f9950b3d9a7.json
  ...
```

每条 generation record 使用已归档 Phase G monitor 的字段：`prompt`、`prompt_id`、`seed`、`weight`、`checkpoint_SHA256`、`initial_noise_SHA256`、21 项 `renoise_trace`、`latent_finite`，以及 `video.{SHA256,RGB_SHA256,lossless}`。
也支持其 `matrix_row` 中的 prompt/seed/weight。
原始记录不需要修改或重新保存 checkpoint；路径不同只改配置。

默认要求记录齐全、真实 checkpoint hash 匹配、两臂 initial noise 和全部 renoise traces 匹配。
确实只有原片且没有记录时，可显式设置 `require_generation_records=false` 和 `record_pattern=null` 做**未验证生成溯源的诊断评分**；报告会标 `UNVERIFIED`，不能将其冒充严格配对复现。
RGB pixel format 检查本身不证明 lossless，默认还要求原 generation record 的 lossless roundtrip 证据及视频 SHA256。

## 自动运行

```bash
# CPU：ffprobe 检查 832×480 / 81帧 / 16fps / RGB，校验原片和配对记录，构建所有任务。
PYTHONNOUSERSITE=1 python -B evaluation/run.py prepare --config outputs/evaluation.local.json

# GPU：明确选择已分配的物理卡；脚本启动每个评分进程前会检查该卡空闲。
# 将 4 换成实际获分配的卡，或直接传完整 GPU UUID。
PYTHONNOUSERSITE=1 python -B evaluation/run.py run --config outputs/evaluation.local.json --gpu 4

# CPU：仅重建报告，无 GPU、无评分、无权重访问。
PYTHONNOUSERSITE=1 python -B evaluation/run.py report --config outputs/evaluation.local.json
```

`run` 自动执行 prepare、逐 arm/seed/dimension 官方 compute 函数、配对汇总。
一次只用一张显式指定的空闲卡，每个维度独立进程，结束释放显存；不后台等待、不占用其他用户卡。
每次启动前查询 nvidia-smi，要求 util≤5%、free≥12GiB、无其他 compute 进程；不 kill 任何进程。
所有评分子进程强制 `PYTHONNOUSERSITE=1`，关闭联网，不调用自动下载的 `init_submodules`。
缺环境/资产/依赖时保存日志后停止，绝不自动安装、升级或下载。

VBench 0.1.5 和各维度源码 SHA 必须匹配既有验证版本。`metric_assets.json` 保存与已验证 `init_submodules(local=True)` 等效的本地参数及资产哈希；只读取当前维度所需的评测资产。**这些评测模型权重也未上传 Git**，以后单独准备一次，复用本地缓存即可。
当前源码检查覆盖各 compute 文件；依赖版本、CUDA/GPU变化仍可能带来数值差异，实际运行环境全部记入结果，不能据 CPU 测试宣称新集群已通过 GPU 资格。

## 输出与恢复

```text
<output>/manifest.json                 输入、config/contract hash、视频hash、pairing
<output>/jobs/<arm>/<seed>/<dim>/
    input.json                         官方 membership 与完整 auxiliary metadata
    eligible_full_info.json
    stdout.log                         完整输出/失败 traceback
    native_result.json                 未修改的 VBench 原始结果
    result.json                        统一 per-video 记录
    COMPLETED.json                     input/result/source/assets hash、环境、耗时、VRAM
<output>/reports/<timestamp>/
    report.md
    summary.json
    per_video.csv
```

重跑 `run` 只跳过 hash 匹配的完整任务。不同配置必须新建输出目录；残缺评分产物保留并停止供审查，不覆盖、不删除、不因结果差重跑。未完成任务的报告明确标 PARTIAL。
Human Action 使用官方原 prompt basename `A person is ...-0.mp4` 的 hardlink，保持原片字节/inode不变，避免把 prompt-id 误当动作标签。其他维度不改文件名。

## 分数与结论边界

- 只对每个 prompt 的官方维度 membership 评分，保留 auxiliary 信息；无覆盖维度跳过。
- 保留原始 per-dimension aggregate；另报 raw unit score×100。MUSIQ 的 per-video 原始分数为0..100，统一时只除一次100；不修改 VBench normalization。
- 原始绝对分数、每 seed 效应、prompt 内跨 seed 平均的配对效应、bootstrap 95% CI（10000次）、win/tie/loss、双侧 sign test 均保留。tie tolerance 预设为点数1e-8。
- 分析单位是 prompt，不能把 prompt×seed 当独立样本。这里的检验是未作 multiplicity correction 的描述性输出，不是新预注册成功门；不推断 noninferiority/equivalence 或训练随机实现稳定性。
- GRIT 可能遗漏样本，明确列出 missing，不补零、不把 retained 数当全覆盖。
- **不输出小面板的 Official Total/Quality/Semantic**，也不自动运行完整944评测、视频生成、训练或机制实验。

## 本次交付验证

CPU/数据/路径/配对/报告测试已通过；没有执行 GPU 评分，没有新模型生成视频，没有上传任何权重。
新集群上的评分环境和评测缓存仍需实际准备/验证。原生产模型、KV、State、scheduler、DMD和训练合同未修改。
