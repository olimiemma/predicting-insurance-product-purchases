# Insurance Product Purchase Prediction: A Comprehensive Logistic Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Business Insights](#business-insights)
- [Technologies Used](#technologies-used)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Project Overview

This project develops a **binary logistic regression model** to predict customer purchase behavior for a new insurance product. Using a dataset of 14,016 customers with demographic, behavioral, and transactional features, the analysis identifies key predictors of purchase likelihood and provides actionable marketing insights.

### Business Problem
An insurance company launched a new product and needs to identify which existing customers are most likely to purchase it. With limited marketing budgets, targeting the right customers is critical for maximizing ROI and conversion rates.

### Solution
A predictive model achieving **75.6% accuracy** and **0.827 ROC-AUC** that identifies high-value customer segments and reveals counterintuitive patterns in purchase behavior.

---

## 🔍 Key Findings
<img width="1785" height="1189" alt="download (3)" src="https://github.com/user-attachments/assets/a7fabcdd-8bcb-4f39-90c2-c391942817d1" />

### Surprising Discoveries

1. **The Loyalty Paradox**: "Unclassified" loyalty customers (loyalty = 99) showed the **highest purchase rate (55.5%)**, nearly double that of the most loyal classified customers (24%)

2. **Product Substitution Effect**: Customers **without** existing products were **2x more likely** to purchase:
   - No Product A: 60% purchase rate vs. 28% for Product A owners
   - No Product B: 64% purchase rate vs. 29% for Product B owners

3. **Age as Top Predictor**: Purchasers averaged **39.7 years** vs. **33.0 years** for non-purchasers, with age being the second strongest predictor after spending patterns

4. **Spending Power Dominates**: Product A turnover emerged as the **single strongest predictor** (coefficient = 6.40), suggesting high-value customers are most receptive

### Model Performance

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| Accuracy | 76.2% | 75.6% |
| Precision | 74.8% | 74.9% |
| Recall | 67.4% | 64.9% |
| F1 Score | 70.9% | 69.6% |
| ROC-AUC | 0.833 | 0.827 |

**Result**: Minimal overfitting with excellent generalization to unseen data.

---

## 📊 Dataset Description

### Overview
- **Total Observations**: 14,016 customers
- **Target Variable**: Binary (0 = No Purchase, 1 = Purchase)
- **Purchase Rate**: 42.9% (well-balanced for classification)
- **Training Split**: 70% (9,811 observations)
- **Test Split**: 30% (4,205 observations)

### Features

#### Demographic Features
- `age`: Customer age in years (18-102)
- `age_P`: Partner age in years
- `LOR`: Length of relationship in years
- `lor_M`: Length of relationship in months

#### Behavioral Features
- `loyalty`: Loyalty level (0, 1, 2, 3, 99 = unclassified)
- `prod_A`: Binary indicator of Product A ownership
- `prod_B`: Binary indicator of Product B ownership
- `type_A`: Product A type classification (0, 3, 6)
- `type_B`: Product B type classification (0, 6, 9)

#### Financial Features
- `turnover_A`: Product A spending amount
- `turnover_B`: Product B spending amount

#### Administrative Features
- `city`: City code (65 unique values, 98% concentration in city=2)
- `contract`: Contract type (constant = 2, no variation)
- `ID`: Customer identifier

### Data Quality Issues Addressed
- Invalid age values (24 customers with age < 18) corrected to minimum valid age of 18
- Extreme right skewness in continuous variables addressed through log transformations
- Perfect multicollinearity between age/age_P and LOR/lor_M resolved by removing redundant variables
- Categorical variable `loyalty` properly one-hot encoded to avoid false ordinal assumptions

---

## 📁 Project Structure

```
insurance-purchase-prediction/
│
├── data/
│   └── insurance_data.csv              # Original dataset (not included due to privacy)
│
├── notebooks/
│   └── insurance_purchase_analysis.ipynb    # Main analysis notebook
│
├── outputs/
│   ├── figures/
│   │   ├── eda_distributions.png
│   │   ├── correlation_heatmap.png
│   │   ├── model_comparison.png
│   │   └── roc_curves.png
│   │
│   └── models/
│       └── final_model.pkl              # Saved trained model
│
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── LICENSE                             # MIT License
```

---

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/insurance-purchase-prediction.git
cd insurance-purchase-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Running the Analysis

1. **Open Jupyter Notebook**
```bash
jupyter notebook
```

2. **Navigate to the notebook**
```
notebooks/insurance_purchase_analysis.ipynb
```

3. **Run cells sequentially**
The notebook is organized in sections:
   - Data Loading & Overview
   - Exploratory Data Analysis (EDA)
   - Data Preparation
   - Feature Engineering
   - Model Development
   - Model Evaluation
   - Business Insights

### Quick Start Example

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/insurance_data.csv')

# Prepare features (after preprocessing)
X = df[['age_log', 'LOR_log', 'prod_A', 'prod_B', 
        'turnover_A_log', 'turnover_B_log',
        'loyalty_1', 'loyalty_2', 'loyalty_3', 'loyalty_99']]
y = df['TARGET']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

#### Univariate Analysis
- Examined distribution of all variables
- Identified skewness in continuous variables (age, LOR, turnovers)
- Detected data quality issues (invalid ages, constant variables)
- Assessed target variable balance (43% purchase rate)

#### Bivariate Analysis
- Analyzed relationship between each predictor and TARGET
- Discovered counterintuitive patterns:
  - Product ownership **negatively** correlated with purchase
  - Unclassified loyalty showed **highest** purchase rates
  - Age showed strong **positive** relationship

#### Correlation Analysis
- Identified perfect multicollinearity (r = 1.0):
  - `age` and `age_P`
  - `LOR` and `lor_M`
- Confirmed independence of other predictors (all r < 0.30)

### 2. Data Preparation

#### Variable Removal
Dropped 7 variables:
- `ID`: Customer identifier (no predictive value)
- `contract`: Zero variation (constant = 2)
- `city`: 98% concentration in one city, invalid codes
- `age_P`: Redundant with age
- `lor_M`: Redundant with LOR
- `type_A`: Redundant with prod_A
- `type_B`: Redundant with prod_B

#### Data Quality Corrections
- Fixed 24 invalid age records (age < 18) by setting to minimum valid age of 18
- No missing values detected

#### Feature Transformations
**Log Transformations**: Applied to address right skewness
- `age` → `age_log` (skewness: 1.18 → 0.52)
- `LOR` → `LOR_log` (skewness: 1.11 → 0.22)
- `turnover_A` → `turnover_A_log` (skewness: 21.22 → 2.64)
- `turnover_B` → `turnover_B_log` (skewness: 8.13 → 4.15)

**One-Hot Encoding**: Applied to categorical variable
- `loyalty` (values: 0, 1, 2, 3, 99) → `loyalty_1`, `loyalty_2`, `loyalty_3`, `loyalty_99`
- Reference category: `loyalty_0`
- **Critical correction**: Initial models incorrectly treated loyalty as continuous numeric variable, causing poor performance

### 3. Model Development

#### Three Models Tested

**Model 1: Full Model with Log Transformations**
- Features: age_log, LOR_log, prod_A, prod_B, turnover_A_log, turnover_B_log, loyalty dummies
- Purpose: Maximum predictive power using optimal transformations
- **Result**: BEST performance (76.2% CV accuracy, 0.833 ROC-AUC)

**Model 2: Parsimonious Model**
- Features: age_log, prod_A, prod_B, loyalty dummies
- Purpose: Simpler model for interpretability
- Result: 71.2% CV accuracy (5% lower than Model 1)

**Model 3: Original Scale Model**
- Features: Original untransformed variables (age, LOR, turnover_A, turnover_B, prod_A, prod_B, loyalty dummies)
- Purpose: Test value of log transformations
- Result: 75.7% CV accuracy (coefficient instability due to scale differences)

#### Model Selection Criteria
- Cross-validation performance (5-fold)
- ROC-AUC for discrimination ability
- Generalization to test set
- Coefficient interpretability
- Alignment with EDA findings

**Selected Model**: Model 1 (Full Log-Transformed)

### 4. Model Evaluation

#### Training Performance
- 5-fold cross-validation accuracy: 76.2% (± 0.6%)
- Stable performance across folds (std = 0.0064)
- ROC-AUC: 0.833

#### Test Performance
- Accuracy: 75.6% (minimal overfitting)
- Precision: 74.9% (75% of predicted purchasers actually buy)
- Recall: 64.9% (captures 65% of actual purchasers)
- F1 Score: 69.6%
- ROC-AUC: 0.827

#### Confusion Matrix (Test Set)
|  | Predicted No | Predicted Yes |
|--|--------------|---------------|
| **Actual No** | 2008 (TN) | 392 (FP) |
| **Actual Yes** | 633 (FN) | 1172 (TP) |

- **Specificity**: 83.7% (correctly identifies non-purchasers)
- **False Positive Rate**: 16.3%
- **False Negative Rate**: 35.1%

---

## 💼 Business Insights

### Target Customer Profile

#### High-Priority Segments

**Segment 1: High-Value Prospects**
- High Product A spending (top quartile)
- Do NOT own Products A or B
- Age 35-45
- **Predicted conversion rate**: 70-80%

**Segment 2: Unclassified Loyalty Gold Mine**
- Loyalty status = 99 (unclassified)
- Likely newer customers (shorter LOR)
- Represents 50% of customer base
- **Actual conversion rate**: 55.5%

**Segment 3: Mature Demographics**
- Age 38+
- Higher income indicators (high turnover)
- **Predicted conversion rate**: 60-70%

### Marketing Recommendations

1. **Prioritize High-Spending Non-Owners**
   - Target customers with high Product A turnover but no product ownership
   - Expected ROI: 2-3x higher than random targeting

2. **Invest in Unclassified Segment**
   - This "mystery" segment drives 64% of all purchases
   - Likely represents customers in active shopping/onboarding phase
   - Increase touchpoints during first 12 months of relationship

3. **Product Positioning Strategy**
   - For **non-owners**: Emphasize comprehensive, all-in-one coverage
   - For **existing owners**: Position as consolidation/upgrade opportunity
   - Message: "Simplify your coverage, reduce costs"

4. **Age-Based Targeting**
   - Focus campaigns on 35-50 age bracket
   - Tailor messaging to life stage needs (family protection, asset coverage)

5. **Resource Allocation**
   - Model precision of 75% means efficient marketing spend
   - Expected waste: Only 1 in 4 targeted customers won't convert
   - Compare to baseline: 57% wouldn't convert with random targeting

### Expected Business Impact

**Baseline vs. Model-Driven Targeting**:
- Baseline accuracy (majority class): 57.1%
- Model accuracy: 75.6%
- **Improvement**: 32% better than random targeting

**Cost Savings Projection**:
- Assume $50 cost per customer contacted
- 14,000 customers × $50 = $700,000 total cost for mass campaign
- With model: Target top 50% (7,000 customers) = $350,000
- Capture 80% of purchasers (4,813 out of 6,016)
- **Savings**: $350,000 with minimal revenue loss

---

## 💻 Technologies Used

### Core Libraries
- **NumPy** (1.21+): Numerical computing and array operations
- **Pandas** (1.3+): Data manipulation and analysis
- **Scikit-learn** (1.0+): Machine learning algorithms and evaluation metrics
- **SciPy** (1.7+): Statistical functions and transformations

### Visualization
- **Matplotlib** (3.4+): Static visualizations
- **Seaborn** (0.11+): Statistical data visualization

### Development Environment
- **Jupyter Notebook**: Interactive development and documentation
- **Python** (3.8+): Programming language

### Key Algorithms & Techniques
- **Logistic Regression**: Primary predictive model
- **One-Hot Encoding**: Categorical variable transformation
- **Log Transformation**: Skewness correction
- **Stratified K-Fold Cross-Validation**: Model validation (k=5)
- **Train-Test Split**: Hold-out validation (70-30 split)

---

## 📚 Lessons Learned

### Critical Mistakes and Corrections

#### 1. Categorical Variable Encoding Error

**The Mistake**: 
Initially treated `loyalty` (values: 0, 1, 2, 3, 99) as a continuous numeric variable in the model.

**Why It Was Wrong**:
- Assumed ordinal relationship: 0 < 1 < 2 < 3 < 99
- But 99 = "unclassified," not "extremely loyal"
- Model couldn't capture non-linear category-specific effects
- EDA showed loyalty_99 had HIGHEST purchase rate, contradicting linear assumption

**The Fix**:
One-hot encoded loyalty into separate binary variables (loyalty_1, loyalty_2, loyalty_3, loyalty_99) with loyalty_0 as reference.

**Impact**:
- Model accuracy increased from 71.2% to 76.2%
- Coefficients aligned with EDA findings
- ROC-AUC improved from 0.759 to 0.833

**Lesson**: Always consider whether categorical variables should be treated as ordinal or nominal. Don't assume numeric codes imply meaningful ordering.

#### 2. Transformation Selection

**Initial Approach**: Used log transformation for all skewed variables.

**Problem Discovered**: Turnover variables remained moderately skewed after log transformation (skewness 2.64 and 4.15).

**Better Approach Considered**: Yeo-Johnson power transformation achieved perfect symmetry (skewness = 0.00) but added complexity.

**Final Decision**: Retained log transformations for interpretability and sufficient performance improvement.

**Lesson**: Perfect isn't always necessary. Weigh statistical optimality against interpretability and complexity.

#### 3. Multicollinearity Handling

**Discovery**: Perfect correlation (r = 1.0) between:
- age and age_P
- LOR and lor_M

**Initial Temptation**: Keep both and let regularization handle it.

**Correct Approach**: Dropped redundant variables before modeling.

**Lesson**: Address perfect multicollinearity explicitly rather than relying on algorithmic fixes.

### Process Insights

1. **EDA Drives Modeling**: Patterns discovered in exploration (product substitution effect, loyalty paradox) guided feature engineering and model interpretation.

2. **Iterate on Errors**: When model coefficients contradict EDA findings, investigate assumptions rather than accepting the results.

3. **Validate Generalization**: Test set performance within 2% of training performance confirmed the model learned genuine patterns, not noise.

4. **Communicate Trade-offs**: Model 2 was simpler (7 features) but 5% less accurate than Model 1 (10 features). Different use cases favor different models.

---

## 🔮 Future Work

### Model Enhancements

1. **Non-Linear Models**
   - Test Random Forest, Gradient Boosting, XGBoost
   - Compare performance to logistic regression baseline
   - Assess interpretability trade-offs

2. **Feature Engineering**
   - Create interaction terms (age × turnover, prod_A × prod_B)
   - Develop customer lifetime value (CLV) features
   - Engineer time-based features (seasonality, tenure bins)

3. **Ensemble Methods**
   - Stack multiple model types
   - Weighted voting across models
   - Potential accuracy improvement: 2-3%

### Data Improvements

1. **Additional Features**
   - Household income
   - Family size/composition
   - Geographic segmentation beyond city
   - Competitor product ownership
   - Digital engagement metrics (website visits, email opens)

2. **Temporal Validation**
   - Test model stability over time
   - Assess performance degradation
   - Develop retraining schedule

3. **Loyalty Classification**
   - Investigate why 50% are "unclassified"
   - Develop model to classify loyalty_99 customers
   - Refine loyalty scoring system

### Business Applications

1. **Propensity Score Integration**
   - Deploy model scores to CRM system
   - Automate segmentation and targeting
   - A/B test model-driven vs. traditional campaigns

2. **Real-Time Scoring**
   - API endpoint for live predictions
   - Score new customers at acquisition
   - Dynamic campaign adjustment

3. **Churn Prediction**
   - Apply methodology to churn modeling
   - Identify at-risk customers
   - Proactive retention campaigns

4. **Product Recommendation**
   - Extend to multi-product recommendation
   - Cross-sell optimization
   - Personalized product bundles

### Research Questions

1. **Why is loyalty_99 most receptive?**
   - Conduct qualitative research
   - Survey unclassified customers
   - Understand decision-making process

2. **Product substitution dynamics**
   - Analyze customers who switched from A/B to new product
   - Understand migration patterns
   - Optimize cannibalization vs. growth trade-off

3. **Optimal contact strategy**
   - How many touchpoints before purchase?
   - Best channel mix (email, phone, direct mail)?
   - Timing optimization (day of week, time of day)

---

## 🤝 Contributing

Contributions are welcome! This project is intended for educational purposes and portfolio demonstration.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments and docstrings for new functions
- Update README if adding new features or sections
- Include test cases for new functionality
- Ensure notebook runs end-to-end without errors

### Areas for Contribution

- Additional visualizations
- Alternative modeling approaches
- Code optimization
- Documentation improvements
- Bug fixes
- Feature engineering ideas

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary**: You are free to use, modify, and distribute this code with attribution.

---

## 📧 Contact

**Project Author**: [Your Name]

- **GitHub**: [@olimiemma](https://github.com/olimiemma)
- **LinkedIn**: [@olimiemma](https://www.linkedin.com/in/olimiemma/)
- **Portfolio**: [My portoflio](https://linktr.ee/olimiemma)

### Questions or Feedback?

- Open an [issue](https://github.com/yourusername/insurance-purchase-prediction/issues) for bugs or feature requests
- Start a [discussion](https://github.com/yourusername/insurance-purchase-prediction/discussions) for questions or ideas
- Connect on LinkedIn for collaboration opportunities

---

## 🙏 Acknowledgments

- **Dataset**: Provided by [Data Source/Organization Name] (if applicable)
- **Inspiration**: Real-world insurance marketing challenges and customer analytics
- **Tools**: Built with open-source Python libraries maintained by dedicated developer communities
- **Learning Resources**: 
  - Scikit-learn documentation
  - "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani
  - Various online tutorials and Stack Overflow community

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/insurance-purchase-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/insurance-purchase-prediction?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/insurance-purchase-prediction?style=social)

**Lines of Code**: ~1,500  
**Analysis Duration**: 40+ hours  
**Documentation**: Comprehensive  
**Reproducibility**: 100%

---

## 🔗 Related Projects

If you found this project interesting, check out these related repositories:

- [Customer Churn Prediction](https://github.com/yourusername/churn-prediction)
- [Credit Risk Modeling](https://github.com/yourusername/credit-risk)
- [Marketing Analytics Dashboard](https://github.com/yourusername/marketing-dashboard)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Made with ❤️ and Python**

[Back to Top](#insurance-product-purchase-prediction-a-comprehensive-logistic-regression-analysis)

</div>
