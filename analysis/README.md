# 离线结果分析

此目录与冻结推理源码分开，脚本只读取既有实验结果，不训练、不推理、不拟合新策略。

从项目根目录运行：

```powershell
& 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe' analysis/diagnose_budget.py
# 结题报告使用的 10 组案例（5 改善 + 5 退化）
& 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe' analysis/diagnose_budget.py --cases-per-side 5
```

参数：`--cases-per-side N` 控制改善/退化各取多少组（默认 3，与最早的 6 组案例一致）；`--out DIR` 可改输出目录（默认 `results/analysis/budget_diagnosis_20261117/`）；`--dpi` 控制 PNG 分辨率（默认 140）；`--palette N` 把 PNG 量化到 N 色调色板（默认 0 关闭）。仓库内现有图表由 `--cases-per-side 5 --dpi 100 --palette 256` 生成，21 张图从 13.3 MB 降到 3.0 MB。脚本会断言 M1 实际使用的 K 与固定档位结果一致，并核对 30 张原图的配对差值与正式统计报告一致。

输入：开发集报告、正式任务完成状态及其指向的运行目录；图例还需要本地原图、掩码和批量结果 PNG。原图、权重与批量 PNG 按原有规则不入库，完整恢复需遵循项目数据准备说明。

生成器确定性选择按原图平均 LPIPS 改善最前 N 组与退化最前 N 组，保留全部三类掩码；边界裁切仅依赖掩码几何。`diagnosis.md` 为在生成图表后人工核对的解释文档，生成器不覆盖它。重新运行会更新 CSV、PNG、选图清单及输入哈希。

目录中的 `verification.json` 记录当前产物的案例数、图表数、输入哈希数与源码指纹状态；结论与失败分析见[最终实验报告](../docs/最终实验报告.md)。
