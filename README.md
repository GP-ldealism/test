# LanPaint 图像修复

基于现有 LanPaint 模型的本地修复原型。基础界面已完成，当前优先完善模型修复效果与预算策略，并推进正式实验。

## 启动界面

在项目目录的 PowerShell 中运行：

```powershell
.\start_ui.ps1
```

或在安装了 `requirements-lanpaint-direct.txt` 依赖的 LanPaint 环境中执行 `python -m app.ui`。
终端显示本地地址后，在浏览器中打开该地址（通常为 `http://127.0.0.1:7860`）。关闭服务可在终端按 Ctrl+C。

需要已准备的 `third_party/LanPaintBench` 上游源码、`checkpoints/256x256_diffusion_uncond.pt` 权重与可用 NVIDIA GPU；恢复说明见 [上游溯源](docs/上游溯源.md)。

上传图片后，用画笔涂抹待修复区域，或切换为上传黑白掩码（白色修复、黑色保留）。点击“预览修复区域”确认，再点击“开始修复”。默认 K=5，输出为 256×256。结果与配置可在页面下载，并保存至 `results/ui/`。

GPU 请求串行排队；第一版尚不支持运行中取消或自适应预算。无掩码时返回原图，全缺失掩码会提示重新选择。上传图片不展示需要参考答案的质量指标。

项目协作入口：[项目进度](docs/项目进度.md)、[下一步计划](docs/下一步计划.md)、[队友改进意见](docs/队友改进意见.md)。

## 当前验证状态

此前全量测试 106 项通过，ruff 静态检查通过。基础界面已通过一次真实 GPU API 修复与下载验证：K=5、NFE=120、采样约 40.05 秒、峰值显存约 2730 MB。基础界面已由用户确认完成，当前重点转回模型完善；2026-11-11 规则冻结流程完成后全量测试为 133 项通过，ruff 检查通过；2026-11-16 完整开发集已完成，冻结与结果核验通过。

完整开发集现有 **10 张原图、60 个掩码组合、180 条档位结果，0 失败**，六策略共 360 条记录。M1 在与 B2 相同总 NFE=7200 下，开发集 LPIPS 从 0.088518 降至 0.085453（降低 **3.46%**），但 PSNR 与边界 MAE 略差。规则已冻结；正式测试未复现稳定收益，开发集改善不代表泛化结论。

产物：[开发集分析](./results/metrics/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb/development_analysis.md)、[冻结配置](./results/raw/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb/frozen_policy.json)、[质量—成本图](./results/metrics/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb/budget_quality.png)。

正式验证已于 **2026-11-16 23:11（北京时间）** 完成：30 张测试原图、10 张共同扩展子集，主测试加面积扩展和两次新增种子共 **540 次生成、1080 条策略记录，0 失败**，四阶段验收通过。主测试相同 NFE=10800 下，M1 LPIPS 为 0.119657，B2 为 0.117012，M1 均值更差 2.26%，配对差值区间跨零；15% 面积子集有改善，30% 面积三种子合并则退化。详见[正式实验报告](results/metrics/formal_campaign_20261116/formal_campaign_report.md)。当前优先完成案例与失败分析，保留固定 K=5 默认策略；计划见[下一步计划](docs/下一步计划.md)。

## 批量实验可靠性更新（2026-11-11）

结果目录改为 `results/raw/<experiment_id>/seed_<seed>/<run_id>/`，指标采用对应的 `results/metrics/` 目录；重跑保留旧日志并复用匹配缓存。运行报告给出实际输出路径。

面积扩展和重复实验需用 `--baseline-run` 指向已完成的种子 42 主测试输出目录。两者使用主测试预先固定的同一子集：20 图主测试取前 5 图，30 图取前 10 图；命令示例见 `scripts/run_experiment.py`。报告的 `comparison_complete` 表示样本结果完整，`equal_nfe` 表示完整集合上的 B2/M1/A1/C1 实际累计 NFE 相等。

归一化与规则冻结流程已实现：完整开发集运行使用 `--freeze-development --warmup`，成功后在该次运行目录生成 `frozen_policy.json`；测试集必须传入 `--frozen-policy <路径>`。加载时核对源码、权重和采样配置，拒绝开发集图片泄漏。开发集选择结果不代替正式测试结论。

使用 `python scripts/analyze_development.py --run-dir <开发集运行目录>` 导出预算质量图、逐样本收益表和分析 Markdown。
