# PIERS 本机全景分割工作台

本机 FastAPI 网页通过 SSH 调用服务器 GPU。街景运行发布版完整 I1/I2/I3＋开放词汇协同模型，默认禁用尚未全量验收的候选特征缓存。室内目前提供通用 DINO/SAM2 文本分割，明确标注未做室内 LoRA 适配。

## 启动

Windows 安装 Python 3.12 后，在仓库执行：

```powershell
python -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r apps/panorama_web/requirements.txt
.\.venv-web\Scripts\python.exe apps/panorama_web/app.py --port 8010
```

浏览器打开 http://127.0.0.1:8010 。也可用 `apps/panorama_web/start.ps1`，它使用独立轻量环境，不需要本机安装 torch 或下载GPU模型。默认只监听本机，不暴露到公网。

进入“连接设置”，填写当前 SSH 地址、端口、用户名、密码，并检查高级路径。密码只存在本机服务的进程内存中，重启后需重新输入；也支持现有 SSH 私钥/agent 或 `PIERS_SSH_KEY`。非敏感连接设置、SSH主机指纹、图片、任务和结果存入 `%LOCALAPPDATA%/PIERSWorkbench`，可通过 `PIERS_WEB_HOME` 改到其他盘。不要把运行数据提交 Git。

服务器先按仓库 `REPRODUCE.md` 准备依赖和权重，环境需有 torch、transformers、Pillow、numpy、opencv 和 SAM2 源码。高级设置的推理仓库必须是包含 `scripts/piers/full_model_fusion_v3.py` 和 `experiments/paper20261109/runtime/paper_paths.py` 的发布版 checkout。网页将少量服务脚本上传到独立运行目录，不改推理仓库或权重，不启动训练。

正式展示建议单独 clone 一份推理代码，例如 `/root/autodl-tmp/piers-web-frozen-v03`，冻结到 `82d20b5b4ba24c42666dd8fee548d5c8618505c8`，与效率开发 checkout 分开；权重可以通过 `PIERS_REMOTE_WEIGHTS` 共用已校验的只读文件。当前网页默认此独立源码目录。后续更换已验收的效率版本应显式更新模型配置及其指标，不能继续套用旧版分数。

可选环境变量：`PIERS_SSH_HOST`、`PIERS_SSH_PORT`、`PIERS_SSH_USER`、`PIERS_SSH_KEY`、`PIERS_REMOTE_REPO`、`PIERS_REMOTE_PYTHON`、`PIERS_REMOTE_WEIGHTS`、`PIERS_REMOTE_RUNTIME`、`PIERS_AGENT_PORT`。密码环境变量 `PIERS_SSH_PASSWORD` 仅用于受控启动会话，不写 `.env` 或仓库。

## 使用与输出

- 上传 PNG/JPEG/WebP（30MB以内，3600万像素以内），选择室外街景或室内。
- 文本留空：使用可扩展的场景默认类别表。街景21类，包含 curb、crosswalk、truck、bus 等；室内8类。
- 文本非空：输入逗号分隔的英文类别/短语或已收录的常见中文类别。未知中文会提示使用英文，不悄悄删词；复杂关系指令的准确执行尚未验证。
- 原图/语义/实例/滑动对照，透明度、缩放拖动、点击类别/实例突出显示；历史任务可再次打开。
- ZIP 中 `classes.png` 是0..N−1类别ID、255未知；`instances.png` 是16位全图实例ID、0背景。`semantic.png`/`instance_layer.png` 是透明掩码；JPG 是叠加可视化；JSON 保存类别、实例及真实运行审计。
- 跨 FoV 关联要求共同可见区域的重叠证据，处理水平环绕；同类相邻实例不会仅因相邻而合并。无重叠碎片可能仍保留为多个实例。此功能未测实例AP，计数是模型估计。
- 普通上传图无 GT，不显示伪造的 mIoU。可上传与原图同尺寸的单通道PNG计算单图指标，类别ID须对应当前图例；单目标也支持二值标注。GT只在本机评分，不发送给模型。

街景几何使用冻结2048×400协议，输出映射回原图。室内通用文本路线使用2048×1024。不同几何/类别表的交互结果不能直接套用论文指标。“实验对比”展示发布版四个口径的真实历史精度和小样本效率，含性能限制，不把本次上传结果混入主表。

## GPU 协调与停止

私有 GPU 服务只监听服务器127.0.0.1:8766，HTTP通过SSH通道访问，无需开放云端Web端口。模型按需加载、任务串行，空闲90秒自动卸载；连接设置可手动释放。取消在当前计算阶段结束后生效。

网页持有 `/root/autodl-tmp/.piers-gpu-exclusive.lock` 的文件锁，并检查无关GPU计算进程。正式效率测量应先停止提交网页任务、点击释放模型，确认 GPU 没有其他计算进程。实验脚本可以使用同一个 `fcntl.flock(LOCK_EX)` 互斥；网页会等待锁和 GPU 空闲。独立测量才能用于论文效率结论。

检测结合 NVML 进程列表、`/proc` 的 NVIDIA 设备占用以及 `reproduce.py run` 等实验调度进程，保守等待整个实验结束。不能仅凭个别容器里不完整的 `nvidia-smi` 输出判断空闲。网页不会停止其他实验进程。

服务器释放接口（任务运行时返回409）：

```bash
curl -s -X POST http://127.0.0.1:8766/api/models/unload
```

本机停止：停止运行 `app.py` 的进程。云端服务空闲卸载后仅保留轻量控制进程；若需彻底停止，只终止独立运行目录 `state/agent.pid` 对应且命令为本应用 `remote_agent.py` 的进程。

## 验证

```powershell
.\.venv-web\Scripts\python.exe -m pip install httpx==0.28.1
.\.venv-web\Scripts\python.exe -m unittest discover -s apps/panorama_web/tests -v
```

CPU测试覆盖开放词汇输入、无文本模式、跨视窗和接缝实例关联、错误标签拒绝、指标计算和API契约。测试中的人工掩码仅用于几何/指标单元测试，不用于演示或冒充模型预测。真实GPU验收单独记录。
