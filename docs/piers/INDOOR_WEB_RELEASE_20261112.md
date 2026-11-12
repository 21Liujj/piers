# 室内实验与网页验收记录：2026-11-12

本次正式结果为 `INDOOR-MATCHED-DINO-20261112`。同一Stage1 checkpoint、同一rank4 LoRA和同一8类解码器，373张heldout：原始OmniSAM37.6424%mIoU，I2+I3核心42.0019%，完整I1/I2/I3为46.5591%，差异+8.9167个百分点。正式I1使用Grounding DINO+SAM2，未使用CLIP或Stage2文本checkpoint。完整方法的可靠性校准器在训练区域calib104拟合；这项额外监督用途需要披露。

## 本次纳入的工作

- 室内训练、数据准备及历史诊断脚本，以及今天修正的同权重正式推理、真实I1校准/融合/评分和复现入口。
- 373张三种配置的逐图整数混淆矩阵、正式汇总表、推理清单、四个训练/测试划分、校准器及原始训练loss日志。
- checkpoint/LoRA指纹、GT支持一致性、无越界预测、I2真实执行、推理不读取GT等核验记录。
- 四幅真实数据图的PNG/SVG：主指标、逐类IoU、训练曲线、固定索引定性对比。
- 网页室内适配路由、输出与实例展示、历史记录和含室内373张结果的实验对比页。
- 三张固定heldout图像的真实网页任务，9张逐图截图和1张实验页截图，全部按原目录格式导出。三张类别ID预测与正式完整模型结果逐像素一致。

室外已有成果与冻结v0.3表格保留。新的室内主表不替换室外权重或改变室外实验协议。历史室内六arm、原生13类街景到SPAN8的错误映射以及CLIP对齐试验不是本次主结果；旧准备文档已加历史状态提示。

## 验收入口

```bash
python scripts/piers/verify_indoor_results.py
python scripts/paper/reproduce.py verify-ledgers
python -m unittest discover -s apps/panorama_web/tests -v
```

室内4个协议测试位于 `tests/piers/test_indoor_matched_protocol.py`，覆盖独立FoV投票、可靠性保护、有效修正和指标/越界标签计数。网页既有18个CPU契约测试覆盖交互API、导出、GT评分、GPU协调与跨视窗实例关联。完整性与发布前检查记录位于 `results/indoor_matched20261112/publication_validation.json`；正式GPU结果核验位于 `FINAL_VERIFICATION.json`。

## 存放与结论边界

小型证据与代码保存在Git；186MB共享权重、完整逐图证据、网页案例及截图包保存在 [v0.4 Release](https://github.com/21Liujj/piers/releases/tag/v0.4-indoor-20261112)，大小与SHA256索引在 `assets/indoor20261112/release_assets.json`。[复现说明](../../REPRODUCE_INDOOR.md)包含下载、CPU核验、GPU网页初始化和重跑入口。SSH密码、令牌、个人运行配置及外部大数据集不纳入Git。

室内为监督跨场景适配；室外为冻结源域权重的跨数据集迁移。室内只完成一折、一组训练种子，fit/calib房间隔离未核验，heldout曾用于smoke诊断；LoRA未宣称全局最佳。门、沙发与窗仍有明显弱项。网页实例数量不是实例AP；交互耗时含模型加载及传输，也不能替代论文隔离时延测量。
