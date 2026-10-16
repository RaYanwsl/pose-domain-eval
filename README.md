# 三维人体姿态跨域泛化评测

合成域上的误差不能直接当成真实拍摄的结果。这个仓库用同一套 24 关节、同一种根节点对齐和同一种相似对齐，比较源域和目标域的 MPJPE、PA-MPJPE，并单独统计脚滑和朝向跳变。

样本按室内、室外、遮挡、着装、远距离分组。仓库里的姿态是可复现夹具，不是 THuman 扫描，也不是训练出来的估计器。把真实预测导出成同样的 `(T, 24, 3)` 数组后，可以套同一份汇总。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pose_eval evaluate --out outputs/report
pytest -q
```

报告在 `outputs/report/report.md`。室外样本带全局尺度误差，所以 MPJPE 高、PA-MPJPE 低。遮挡样本的误差集中在上肢。室外行走夹具用来暴露脚滑和朝向跳变，这两项不会出现在单帧平均里。
