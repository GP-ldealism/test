# 正式实验运行辅助脚本

这些文件是 `.cache/` 中实际运行辅助脚本的留档副本。运行时须放回项目根目录下的 `.cache/`，因为脚本按该位置解析项目根目录；不要直接从本目录运行。

- `preflight.py` 对应 `.cache/formal_preflight_20261116.py`。
- `main_runner.py` 对应 `.cache/formal_main_20261116.py`。
- `start_main.ps1` 对应 `.cache/start_formal_main_20261116.ps1`。
- `followup_runner.py` 对应 `.cache/formal_followup_20261116.py`。
- `start_followup.ps1` 对应 `.cache/start_formal_followup_20261116.ps1`。

主测试状态见 `results/main_test_20261116_status.json`。后续状态见 `results/formal_followup_20261116_status.json`，`waiting_main` 表示等待主测试验收、尚未执行扩展推理。后续顺序固定为面积扩展、种子 2026、种子 3407；任一阶段失败停止后续任务。

不要在现有主测试进程运行时再次启动主测试。主测试意外退出后，确认原进程已结束，再运行原主测试脚本即可复用匹配缓存；后续等待进程若已记录失败，先确认该进程退出，再启动原后续脚本。后续脚本用进程文件锁避免同时启动两份，恢复时核验已完成报告并复用。

源码与输入哈希、完整性或 NFE 验收失败时应先定位原因，不修改冻结文件绕过检查。所有脚本均使用既有冻结推理源码；辅助分析位于源码指纹范围之外，避免使已冻结规则失效。
