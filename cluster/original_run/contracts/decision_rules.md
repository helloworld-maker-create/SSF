# 冻结判定与解释边界（待批准）

## Baseline与人工门

固定健康rubric见baseline_health_gate.json；T=0必计严重时序失败，严重类别按视频并集不重复计数。
Baseline600完成后必须导出64条去身份完整视频引用及空白表，封存AI/模型映射，状态AWAITING_USER_HEALTH_REVIEW，释放GPU、销毁PG并退出。
无真实冻结用户评分及明确阶段C继续批准，不启动SSF。不允许AI制造用户分数、批准或健康PASS。
发布EMA同prompt也失败只标参照/任务适用性问题；保留样本与阈值，不通过事后改门放行。
view-only仅源片不兼容时统一转码，保持帧数/fps/分辨率，标明源SHA及实际观看版本；只评分原lossless科学输入。

## MVP

唯一主要终点：600 EMA的SSF−Baseline主体一致性。raw/raw次级，不事后换赢家或checkpoint。
有意义改善提案>=0.5 raw×100点。两推理seed先在prompt内平均；10000 prompt-bootstrap、mean/median/SE/95%CI、按1e-6点tie容差计算W/T/L及逐seed方向。
报告sign test时明确其estimand为非tie符号概率，与均值CI不同；不把四舍五入的0判为tie。
次级探索性报告，不要求16维全部显著；不将tiny panel无下降视作正式非劣效或等效。
风险提案：HumanAction/DD下降10点、smooth下降1点、MUSIQ下降2点、text-video下降1点，结合两seed同向及冻结盲评。DD不等于动作质量；frame proxy为0..255变化，可能混合镜头/局部运动/光度。

依次判定：
1. 缺测/审查不足/关键未解决不确定性 -> INCONCLUSIVE。
2. 明显动作/成像/文本代价 -> TRADEOFF，同时报告有没有收益。
3. 主体均值>=.5，均值95%CI下界>0，两seed正，且无明显代价 -> PROMISING_SIGNAL。
4. 主体CI同时覆盖>=+.5与<=-.5 -> INCONCLUSIVE。
5. 其余有效结果 -> NO_CLEAR_SIGNAL，包括明确负面结果。

一个training pair不提供训练总体稳定性；两个infer seeds不是训练复制，prompt模板相关未建模。
无子集Official Total，不补零造Quality/Semantic。不能把新旧变化唯一归因B64/EMA/工程，也不宣布旧R1/R2必然崩溃。
F路由负结果保留，不作默认方法；不以本MVP穷尽M1/M2/M3解释。

## 授权、失败与停止

Plan冻结不等于科学批准或执行批准；分A/B/C真实用户事件记录，各自预算，未消费额度不能自动挪用。
G1测得超预算 -> BLOCKED_BUDGET；不自动减B64或加预算。
资格工程失败最多一次最小修复/受影响资格重验，仍在48内。
正式状态/EMA/梯度/cache完整性失败停止报告；资源中断只用相同run的COMMITTED恢复，失败与重算计入预算。
有限但差的质量结果不重试，不删负结果。生成技术失败最多一次同权重/tape的新attempt，计入800，绝不因质量重生成。
Baseline FAIL/INCONCLUSIVE停止本Goal。完成合法MVP即停止，不自动50/100曲线、30s或full944。
