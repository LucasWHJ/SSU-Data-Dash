# Data Dash 模型修改与验证报告

日期：2026-10-20。此文件记录本次本地改动与实验结果，不是比赛要求提交的四页 PDF。没有提交或推送到 GitHub。

## 1. 方案与当前使用的模型

`run.py` 的 `validate` 和 `predict` 都调用 `files/models.py` 中的 `make_model()`。现在两个赛道都启用 scikit-learn 的 `HistGradientBoostingRegressor`，原来的 `ListPriceBaseline` 和 `NearestSalesBaseline` 仍保留，便于对照。

- **Off-market**：仅从预测时有的房屋、位置、挂牌日期与文字描述提取特征；拟合 `log(sold_price)`。没有读取 `list_price`。
- **List-aware**：使用同一组特征，再加入挂牌价及每平方英尺挂牌价；拟合 `log(sold_price / list_price)`，预测时再乘挂牌价。
- 类别特征包括房型、房屋风格、市镇和社区。类别清单只从本轮训练数据建立，预测时沿用，未知类别视作缺失值。
- 两个模型都用 260 次迭代、学习率 0.06、最多 31 个叶节点、最小叶节点样本数 40、L2 正则 0.1。关闭模型内部随机留出，验证切分只由原有逐月流程控制。

## 2. 具体代码改动

| 文件与行号 | 改动 | 原因 |
|---|---|---|
| `files/features.py:9-20` | 定义四个类别字段和六组固定的描述关键词。 | 表达房屋类型、区域及状态；关键词不从未来或测试集学习。 |
| `files/features.py:23-28` | 将单个面积或区间面积转换成数值。 | 统一处理 `house_area`。 |
| `files/features.py:31-50` | off-market 特征加入车库、地块尺寸及面积、挂牌时间与季节、社区等类别，以及描述标记。 | 利用挂牌时已知的信息；没有使用挂牌价或成交后字段。 |
| `files/features.py:53-58` | list-aware 额外加入挂牌价及每平方英尺挂牌价。 | 利用这个赛道允许的挂牌价。 |
| `files/models.py:9-15` | `make_model()` 为两条赛道返回新的 `FeatureModel`。 | 让现有验证与预测命令实际使用改进模型。 |
| `files/models.py:18-52` | 保留两个原有基线模型。 | 方便日后复核与比较。 |
| `files/models.py:55-98` | 配置梯度提升树，固定随机种子，仅从训练数据确定类别，按赛道分别定义对数目标并还原预测价格。 | 处理非线性关系、未知类别和两赛道不同的预测目标。 |
| `check_submission.py:51` | 将旧的 `import data` 改为 `from files import data`。 | 修复目录重组后 `python run.py check` 无法运行的问题。 |
| `tests/test_starter.py:9-11` | 将不存在的 `pyfiles` 导入改为当前项目模块路径。 | 恢复原有测试集。 |
| `.gitignore:6-8` | 允许两份最终提交 CSV 出现在 Git 变更中。 | handout 要求最终仓库包含它们；本次没有提交或推送。 |

`run.py` 的时间切分、评分公式和报告生成逻辑没有修改。没有使用 `external_avm_estimate`、隐藏测试成交价或外部地图数据。

## 3. 逐月向前验证结果

验证月份为 2025-12 至 2026-05。每个月只用该月之前已经成交的销售训练，再预测该月的房屋；每条赛道 10,069 套。指标来自本次运行的 `outputs/validation_predictions.csv` 和 `reports/validation_report.json`。

| 赛道 | Handout 起始基线 MdAPE | 本次 MdAPE | 本次 MAPE | MdAPE 的 bootstrap 95% 区间 |
|---|---:|---:|---:|---:|
| Off-market | 20.8% | **9.75%** | 12.39% | 9.48%–10.02% |
| List-aware | 7.3% | **4.23%** | 5.31% | 4.13%–4.33% |

先前本地改进模型留下的结果为 9.83% / 4.22% MdAPE。本次 off-market 低 0.08 个百分点；list-aware 高 0.01 个百分点，基本持平。上述结果只代表已知成交价的验证期，不是 2026 年 6–9 月隐藏测试成绩。多次在同一验证期尝试模型后，最佳验证结果可能偏乐观；也未逐项做消融实验，不能把提升归因于某个单独特征。

| 月份 | Off-market MdAPE | List-aware MdAPE |
|---|---:|---:|
| 2025-12 | 10.12% | 4.02% |
| 2026-01 | 9.44% | 4.06% |
| 2026-02 | 9.65% | 4.26% |
| 2026-03 | 9.24% | 4.21% |
| 2026-04 | 9.85% | 4.23% |
| 2026-05 | 10.03% | 4.52% |

分房型看，off-market 的 Co-Op Apt 为 11.44%、Detached 为 10.68%；list-aware 的 Semi-Detached 为 5.61%、Detached 为 5.41%，都高于该赛道总体 MdAPE。这提示某些房型仍更难估价。报告中的 bootstrap 区间把房屋视作独立样本；同月房屋受共同市场波动影响，实际不确定性可能更大。

## 4. 运行与复现

- 小样本模型检查：两赛道均成功拟合并输出有限价格；off-market 特征不含挂牌价或成交后字段。
- `python run.py validate`、`python run.py report`、`python run.py predict`：均成功。
- `python run.py check`：两份 CSV 均显示 `OK`，off-market 3,974 行，list-aware 4,064 行。
- `python -m pytest -q`：18 项通过。
- `git diff --check`：通过。

本地复现时先设置 `DATA_DIR` 指向项目的 `datasets` 目录，例如在 macOS/Linux：

```bash
DATA_DIR="$(pwd)/datasets" python run.py validate
python run.py report
DATA_DIR="$(pwd)/datasets" python run.py predict
DATA_DIR="$(pwd)/datasets" python run.py check
```

生成的验证报告在 `reports/validation_report.md`，两份预测文件在 `submissions/`。比赛正式 PDF、视频及提交仍需另行完成。
