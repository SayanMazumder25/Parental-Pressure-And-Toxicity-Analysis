# Parental Pressure and Toxicity Analysis

### Exploring academic expectations, emotional experiences, and parental influence through data

An exploratory data analysis and machine learning project investigating students' reported experiences of parental pressure and perceived parental toxicity, using survey responses, statistical analysis, visualisations, and regression models.

---

## Overview

Parental expectations can influence students' academic experiences, personal decisions, emotional well-being, and future aspirations. However, academic pressure and perceived parental toxicity are complex experiences that cannot be understood through examination results alone.

This project explores the relationship between reported parental pressure, perceived parental toxicity, and selected demographic characteristics. It examines experiences such as pressure to achieve higher marks, comparison with other children, parental involvement in decision-making, and the dismissal of children's feelings.

Using a survey dataset of **142 respondents**, the project combines exploratory data analysis (EDA), correlation analysis, demographic comparisons, and machine learning to identify patterns that may warrant further investigation.

**The objective is not to judge individual families, but to understand the patterns reported by participants and explore how data analysis can contribute to a more informed discussion of parental pressure.**

## Research Questions

The project investigates the following questions:

1. What experiences are associated with parental pressure and perceived parental toxicity?
2. Do reported parental-pressure and perceived parental-toxicity ratings vary across gender groups?
3. Do these ratings vary across rural, semi-rural, and urban backgrounds?
4. Do parental-pressure and perceived parental-toxicity ratings differ by educational level?
5. Is age associated with parental pressure or perceived parental toxicity?
6. Can machine learning models predict the overall ratings using the available survey data?

## Dataset

The analysis uses survey responses collected from **142 participants**.

The dataset includes demographic information and self-reported ratings concerning different aspects of parental pressure and perceived parental toxicity.

The variables cover areas such as:

- Age and gender
- Educational level
- Living background
- Academic expectations and examination pressure
- Comparison with other children
- Parental involvement in personal decision-making
- Friendship-related pressure
- Guilt, feelings, and fear
- Overall parental-pressure and perceived parental-toxicity ratings

The dataset is observational and based on self-reported experiences. It should not be interpreted as a representative survey of all students or families in India.

## Technologies Used

- **Python** — data analysis and model development
- **Pandas** — data manipulation and preprocessing
- **NumPy** — numerical operations
- **Matplotlib and Seaborn** — data visualisation
- **Scikit-learn** — regression models and model evaluation
- **Jupyter Notebook** — interactive analysis and documentation
- **Joblib** — saving and loading trained machine learning models
- **Streamlit** — interactive application, if deployed

## Project Workflow

### 1. Data Preparation

The survey data was inspected and prepared for analysis. This included reviewing column names, examining variables, and preparing demographic and rating-based features for statistical analysis and modelling.

### 2. Exploratory Data Analysis

Visualisations were used to examine the distributions of parental-pressure and perceived parental-toxicity ratings. The analysis also explored differences across gender, educational level, living background, and age.

### 3. Correlation Analysis

Spearman correlation was used to investigate associations between individual survey items and the overall ratings.

Some of the strongest observed associations included:

| Survey item | Parental-pressure rating | Perceived parental-toxicity rating |
|---|---:|---:|
| Pressure to achieve higher marks | 0.68 | 0.47 |
| Comparison with other children | 0.62 | 0.43 |
| Parental decision-making | 0.63 | 0.56 |
| Dismissal of children's feelings | 0.22 | 0.61 |
| Career-related pressure | 0.58 | 0.41 |

*Note: Values are Spearman correlation coefficients from the analysed sample. Correlation indicates association, not causation. The table uses the corresponding survey items as analysed in the notebook.*

### 4. Demographic Analysis

The study compared the distributions of ratings across demographic groups using descriptive statistics and visualisations.

Notable observations included:

- Academic expectations, comparison, and parental decision-making showed relatively strong associations with reported parental pressure.
- Dismissal of children's feelings showed the strongest observed association with perceived parental toxicity.
- The semi-rural group recorded the highest average ratings for both outcomes in this sample.
- School students recorded the highest average parental-pressure rating.
- The postgraduate group recorded the highest average perceived parental-toxicity rating, although this group contained only four respondents.
- The observed associations between age and the two outcomes were comparatively weak.

These are descriptive observations from the available sample, not universal conclusions about demographic groups.

### 5. Machine Learning

Four regression approaches were evaluated:

- Dummy Regressor
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The models were evaluated using the coefficient of determination (\(R^2\)) and mean absolute error (MAE).

The selected models were:

- **Linear Regression** for parental-pressure prediction
- **Random Forest Regression** for perceived parental-toxicity prediction

For the parental-pressure model, the reported test-set results were:

| Metric | Result |
|---|---:|
| \(R^2\) | 0.44 |
| Mean Absolute Error (MAE) | 1.32 |

The pressure model's \(R^2\) indicates that it accounted for approximately 44% of the variation in the test-set ratings in that evaluation. The MAE indicates an average absolute prediction error of approximately 1.32 rating points.

These results are preliminary because the dataset is small and the test set contains only 29 observations. They do not establish that the model will perform similarly on new populations or in practical settings.

## Key Takeaways

The analysis highlights several patterns worth further investigation:

- Academic expectations and comparisons with other children are strongly associated with reported parental pressure.
- Parental involvement in decision-making is associated with both parental pressure and perceived parental toxicity.
- The dismissal of children's feelings is particularly strongly associated with perceived parental toxicity.
- The distributions of ratings vary across demographic groups, although small and imbalanced subgroup sizes limit the reliability of these comparisons.
- Machine learning offers a possible way to investigate patterns in survey data, but stronger validation is needed before making reliable predictive claims.

The project emphasises the importance of interpreting survey results carefully and distinguishing statistical associations from causal explanations.

## Limitations

The findings should be interpreted in light of the following limitations:

- **Voluntary convenience sampling:** Participation was voluntary, and the final sample consisted of 142 respondents.
- **Demographic imbalance:** The sample contained 128 undergraduates, 10 school students, and four postgraduates, with unequal gender representation.
- **Self-reported ratings:** Responses reflect participants' perceptions and may be influenced by individual interpretation and experience.
- **Cross-sectional design:** The study identifies associations at a particular point in time and cannot establish causation or changes over time.
- **Small subgroup sizes:** Comparisons involving underrepresented groups may be sensitive to individual responses.
- **Measurement limitations:** Complex family experiences may not be fully captured by a limited set of survey questions.
- **Exploratory machine learning:** The models were trained on 113 observations and evaluated on 29 observations, limiting the reliability of performance estimates.

The results should not be generalised to all students or families without further research.

## Future Work

Potential directions for further development include:

- Collecting a larger and more demographically representative dataset.
- Improving the clarity, coverage, and reliability of survey questions.
- Conducting additional statistical tests and reporting uncertainty around observed differences.
- Exploring participants' experiences through qualitative interviews or open-ended responses.
- Applying cross-validation and appropriate hyperparameter tuning.
- Evaluating models on larger, independent test datasets.
- Checking predictor-target relationships to reduce the risk of target leakage.
- Investigating potential educational and family-support interventions through appropriately designed follow-up studies.

## Running the Project Locally

### Prerequisites

Install Python and Git. A virtual environment is recommended.

### 1. Clone the repository

Replace the placeholder URL with the actual GitHub repository URL after publishing the project.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

Open the notebook in Jupyter:

```bash
jupyter notebook Toxicity.ipynb
```

Run the notebook cells in order. Ensure the dataset paths match the repository structure.

### 5. Run the Streamlit application

If the repository includes a Streamlit application, run it using the correct Python filename:

```bash
streamlit run app.py
```

If the application is still named `Abs.py`, use:

```bash
streamlit run Abs.py
```

The trained model files and required data must be available at the paths expected by the application.

## Ethical Considerations

Parental pressure and perceived toxicity are sensitive subjects. The analysis should be used to explore reported experiences, not to label or diagnose individual parents or families.

Any public release of survey data should respect participant privacy and applicable consent conditions. Identifying information should not be published, and model predictions should not be treated as definitive assessments of a person's family environment.

## Author

**Sayan Mazumder**

This project combines survey-based research, exploratory data analysis, statistical reasoning, and machine learning to investigate a socially relevant question.

- GitHub: [SayanMazumder25](https://github.com/SayanMazumder25)
---

*This project is an exploratory study based on a limited self-reported sample. Its findings are intended to support further investigation and discussion, not to establish universal or causal conclusions.*