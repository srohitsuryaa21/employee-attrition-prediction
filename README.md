# Employee Attrition Rate Prediction
### A Machine Learning Approach to HR Analytics

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Certificate](https://img.shields.io/badge/Google_Data_Analytics-Coursera-4285F4?logo=google&logoColor=white)

> **Capstone Project** — Google Data Analytics Professional Certificate, Coursera  
> **Author:** Rohit Suryaa Saravanan  
> **Programme:** MSc Data Science, Hochschule Fulda, Germany

---

## Abstract

Employee attrition poses significant operational and financial challenges for organisations, with replacement costs estimated at 50–200% of an employee's annual salary. This project presents a comprehensive data-driven framework for predicting voluntary employee turnover using the IBM HR Analytics Employee Attrition & Performance dataset. Employing the Google Data Analytics APPASA methodology (Ask, Prepare, Process, Analyse, Share, Act), the study conducts systematic exploratory data analysis across 35 demographic, behavioural, and organisational features, followed by the development and evaluation of four supervised classification models. On the original IBM data, the best model, logistic regression, reaches a test AUC-ROC of 0.80 (5-fold CV AUC 0.81), ahead of gradient boosting (0.78), random forest (0.77) and a decision tree (0.66). Findings indicate that overtime obligation, compensation level, and composite job satisfaction are the primary predictors of attrition, with actionable HR recommendations derived accordingly.

---

## Dataset

| Property | Detail |
|----------|--------|
| **Name** | IBM HR Analytics Employee Attrition & Performance |
| **Source** | [Kaggle — pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Records** | 1,470 employees |
| **Features** | 35 (demographic, job-related, satisfaction, compensation) |
| **Target Variable** | `Attrition` (binary: Yes / No) |
| **Class Imbalance** | ~16.1% positive (attrition = Yes) |
| **Origin** | Fictional dataset created by IBM data scientists for HR research |

> **Data:** the notebook loads the original dataset from IBM's public GitHub copy (`IBM/employee-attrition-aif360`), so it runs anywhere without a Kaggle login. An earlier version generated a synthetic stand-in whose attrition labels were drawn independently of every feature, which capped every model near AUC 0.5; the results here come from the real data.

---

## Methodology

This study follows the **Google APPASA analytical framework**:

### Phase 1 — Ask
Definition of the research problem, success metrics (AUC-ROC, chosen because only 16.1% of employees left), and stakeholder deliverables. Primary research questions:
- Which employees are most likely to leave, and why?
- What is the probability of departure over the next employment period?
- What organisational interventions would most effectively improve retention?

### Phase 2 — Prepare
Dataset inventory: 35 features assessed across four categories — demographics (Age, Gender, MaritalStatus), job attributes (JobRole, Department, JobLevel), satisfaction indicators (JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance), and compensation (MonthlyIncome, StockOptionLevel, PercentSalaryHike). Three constant-value columns (EmployeeCount, Over18, StandardHours) identified and removed.

### Phase 3 — Process
- Null value and duplicate record verification
- Label encoding of binary categorical variables (OverTime, Gender)
- One-hot encoding of multi-class categorical variables (Department, JobRole, MaritalStatus, BusinessTravel, EducationField)
- Derivation of five engineered features (see Feature Engineering section)
- 80/20 stratified train-test split; StandardScaler applied for Logistic Regression

### Phase 4 — Analyse
Univariate and bivariate EDA across all 35 features. Statistical patterns examined across attrition cohorts using distribution plots, grouped bar charts, box plots, and a full lower-triangular correlation heatmap.

### Phase 5 — Share
Synthesis of findings into an HR insights dashboard and a ranked list of attrition drivers with supporting statistical evidence.

### Phase 6 — Act
Training, cross-validation, and evaluation of four classification models, followed by feature importance analysis and structured HR recommendations.

---

## Feature Engineering

Five new features were derived from existing variables to improve model signal:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `SatisfactionScore` | Mean(JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction) | Composite wellbeing indicator |
| `PromotionGap` | YearsAtCompany − YearsSinceLastPromotion | Captures career stagnation |
| `LoyaltyIndex` | YearsWithCurrManager / (NumCompaniesWorked + 1) | Relative organisational loyalty |
| `TenureToAge` | YearsAtCompany / Age | Seniority relative to career stage |
| `IncomePerYear` | MonthlyIncome × 12 | Annualised compensation for interpretability |

---

## Results (original IBM data)

| Model | Test AUC | 5-fold CV AUC |
|-------|---------:|--------------:|
| **Logistic Regression** | **0.798** | 0.808 ± 0.030 |
| Gradient Boosting | 0.785 | 0.789 ± 0.029 |
| Random Forest | 0.772 | 0.776 ± 0.031 |
| Decision Tree | 0.663 | 0.624 ± 0.041 |

The best model finds 36% of leavers in the test set (17 of 47) at 71% precision. Employees on overtime left at 30.5% against 10.4% without; sales representatives left at 39.8%.

## Models & Evaluation

| Model | Configuration | Evaluation Metrics |
|-------|--------------|-------------------|
| Logistic Regression | max_iter=1000, StandardScaler | AUC-ROC, Odds Ratios, Precision-Recall |
| Decision Tree | max_depth=8 | AUC-ROC, Feature Importance |
| Random Forest | n_estimators=200, max_depth=10 | AUC-ROC, Permutation Importance |
| Gradient Boosting | n_estimators=200, lr=0.05, max_depth=5 | AUC-ROC, SHAP-style importance |

All models evaluated using: **AUC-ROC · Accuracy · 5-fold stratified cross-validation · Confusion matrix · Precision-recall tradeoff analysis**

---

## Key Findings

1. **OverTime** is the strongest single predictor of attrition: 30.5% of employees on overtime left, against 10.4% without
2. **Monthly income** shows a significant inverse relationship with attrition; departing employees earn substantially below the median
3. **Age** is negatively correlated with attrition risk — employees aged 20–35 represent the highest-risk demographic
4. **Sales Representatives** (39.8%) and **Laboratory Technicians** (23.9%) exhibit the highest role-level attrition rates
5. **Composite satisfaction score** reliably discriminates between leavers and stayers across all satisfaction dimensions
6. **Frequent business travel** is associated with about three times the attrition rate of non-travel employees (24.9% vs 8.0%)
7. **Promotion gaps** over three years do *not* raise attrition in this data (13.1% vs 17.0%)
8. **Single employees** churn at a meaningfully higher rate than married or divorced peers

---

## HR Recommendations

**Immediate interventions (0–3 months)**
- Conduct an OverTime policy audit; introduce compensatory leave or pay review mechanisms
- Identify high-risk employees (model score ≥ 0.70) and initiate structured retention conversations
- Review compensation benchmarking for Sales and Laboratory functions

**Short-term initiatives (3–6 months)**
- Establish a formal promotion pipeline with explicit timelines; eliminate gaps exceeding 36 months
- Introduce flexible and hybrid working arrangements to mitigate commute-related attrition
- Deploy annual engagement surveys with mandatory departmental action planning

**Strategic programmes (6–12 months)**
- Develop an internal mobility framework targeting high-potential early-career employees
- Integrate the predictive model into the HRIS as a real-time attrition early-warning system
- Design differentiated retention packages for frequent-travel and client-facing roles

---

## Repository Structure

```
employee-attrition-prediction/
│
├── Employee_Attrition_Prediction_Rohit_Suryaa.ipynb   # Main analysis notebook
├── README.md                                           # This file
└── requirements.txt                                    # Python dependencies
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/rohitsuryaa/employee-attrition-prediction
cd employee-attrition-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Employee_Attrition_Prediction_Rohit_Suryaa.ipynb
```

---

## Citation

```
IBM HR Analytics Employee Attrition & Performance.
Available at: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Accessed: 2024. Fictional dataset created by IBM data scientists.
```

---

## Author

**Rohit Suryaa Saravanan**  
MSc Data Science — Hochschule Fulda, Germany  
Google Data Analytics Professional Certificate — Coursera
