# Telecom Customer Churn Prediction

> **Classical Machine Learning Project · Archived**

基于 Telco Customer Churn 数据集，利用用户基本属性、服务信息、合同信息和费用信息预测用户流失。

项目重点不是单纯追求模型分数，而是完整经历：

**数据分析 → 特征预处理 → Baseline → Logistic Regression → Decision Tree → Random Forest → 阈值优化 → 模型解释 → 错误分析 → 泛化能力分析**

---

## Dataset

| 指标 | 值 |
|---|---:|
| Samples | 7,043 |
| Features | 20 |
| Target | Churn（Yes/No） |
| Churn rate | 26.54% |
| Train / Test | 5,634 / 1,409（分层抽样，`random_state=42`） |

---

## Models

- **Logistic Regression** — 含手写 NumPy 实现，L2 正则化，Pipeline 预处理
- **Decision Tree** — 含手写信息增益/连续切分点，深度实验与过拟合分析
- **Random Forest** — Bagging 集成，OOB 评估，验证集阈值搜索

---

## Final Result

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.7743** | **0.5622** | 0.6765 | **0.6141** | **0.8409** | 0.39 |
| Decision Tree | 0.7509 | 0.5273 | 0.5936 | 0.5585 | — | 0.5 |
| Random Forest | 0.7466 | 0.5164 | **0.7166** | 0.6002 | 0.8190 | 0.30 |

> **最终模型：Logistic Regression（threshold = 0.39）**
>
> 在当前数据和实验设置下取得最高 F1 和 ROC-AUC。Random Forest 的 Recall 更高，但产生更多误报。

多数类 Baseline（永远预测"不流失"）Accuracy = 73.46%，因此仅看 Accuracy 没有意义。

---

## Key Findings

1. **Contract type 和 tenure 是最重要的流失预测因子** — Month-to-month 用户和新用户流失风险显著更高。
2. **Fiber optic 用户表现出更高的预测流失风险** — 在 EDA 和模型系数中均稳定出现。
3. **Decision Tree 极易过拟合** — 树深从 2 增加到 20，训练 F1 从 0.587 升至 0.973，但测试 F1 从 0.587 降至 0.505。
4. **Random Forest 通过降低阈值可将 Recall 提升至 71.66%**，但 Precision 下降至 0.516，误报增多。
5. **Logistic Regression 在综合指标上最优** — F1 = 0.6141，ROC-AUC = 0.8409，且具备完整可解释性。
6. **模型存在性能上限** — 部分用户特征在两类间高度重叠，FN=121、FP=197 均非随机错误。

---

## Project Structure

```
Telecom-Churn-Prediction/
├── data/
│   └── data.csv                      # 原始数据集
├── notebooks/
│   └── 电信流失用户预测.ipynb          # 完整实验 Notebook
├── figures/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_internet_service.png
│   ├── tenure_vs_churn.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── learning_curve.png
│   └── generate_figures.py           # 图表复现脚本
├── results/
│   ├── final_results.csv             # 模型指标汇总
│   └── final_report.md               # 完整中文技术报告
├── src/
│   └── README.md
├── .gitignore
└── README.md
```

---

## Learning Objectives

本项目作为经典机器学习基础项目，覆盖以下核心概念：

- Data preprocessing & Pipeline
- Exploratory data analysis (EDA)
- Logistic Regression（含手写实现）
- Cross-validation
- Regularization (L1 / L2)
- Decision Trees（含手写信息增益）
- Overfitting analysis
- Bagging
- Random Forest
- Threshold optimization（验证集搜索，无数据泄露）
- Error analysis (FP / FN)
- Learning curves
- Model interpretation（系数分析）

---

## How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction

# 2. Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter

# 3. Run the notebook
jupyter notebook notebooks/电信流失用户预测.ipynb

# 4. (Optional) Regenerate figures
cd figures
python generate_figures.py
```

---

## License

MIT

---

*项目已归档，不再新增模型或调参。完整技术报告见 [`results/final_report.md`](results/final_report.md)。*
