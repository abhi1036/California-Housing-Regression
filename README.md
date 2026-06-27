# California Housing Price Prediction

A production-ready Machine Learning project that predicts median house values in California using **Multiple Linear Regression**. This project demonstrates the complete machine learning workflow, from data exploration and model training to deployment with a Flask web application.

---

## Project Overview

This application predicts the **median house value** for a California district based on demographic and housing-related features.

The project was built to understand the complete regression pipeline rather than simply training a model.

The workflow includes:

* Data Exploration (EDA)
* Multiple Linear Regression
* Model Evaluation
* Residual Analysis
* Model Serialization
* Flask Web Application
* Deployment Ready

---

## Dataset

**Dataset:** California Housing Dataset

Source: Scikit-Learn

The dataset contains housing information collected from California districts.

### Features

* Median Income
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

### Target

Median House Value

---

## Exploratory Data Analysis

Performed:

* Dataset overview
* Missing value analysis
* Descriptive statistics
* Feature distributions
* Correlation matrix
* Outlier detection using boxplots

Key observations:

* Median Income has the strongest positive correlation with house value.
* Population and Average Occupancy contain significant outliers.
* Average Rooms and Average Bedrooms exhibit multicollinearity.
* House Value is capped at the upper limit in the dataset.

---

## Model

Algorithm:

**Multiple Linear Regression**

The model was trained using Scikit-Learn's `LinearRegression`.

Train-Test Split:

* Training: 80%
* Testing: 20%

Random State:

* 42

---

## Model Evaluation

| Metric   |  Value |
| -------- | -----: |
| MAE      | 0.5332 |
| MSE      | 0.5559 |
| RMSE     | 0.7456 |
| R² Score | 0.5758 |

### Interpretation

* Average prediction error is approximately **$53,320**.
* The model explains approximately **57.6%** of the variance in house prices.
* This serves as a strong baseline model before applying regularization or ensemble methods.

---

## Project Structure

```text
California-Housing-Regression/

├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── linear_regression.pkl
│   ├── features.pkl
│   └── metrics.pkl
│
├── notebooks/
│   └── california_housing.ipynb
│
├── src/
│   ├── config.py
│   ├── predictor.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── __init__.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── reports/
└── screenshots/
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Flask
* Joblib
* Bootstrap 5

---

## Running the Project

### Clone Repository

```bash
git clone <repository-url>
cd California-Housing-Regression
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py
```

### Start Flask Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## Future Improvements

* Ridge Regression
* Lasso Regression
* Elastic Net
* Hyperparameter Tuning
* Cross Validation
* Docker Support
* CI/CD Pipeline
* Cloud Deployment
* Interactive Model Comparison Dashboard

---

## Learning Outcomes

This project demonstrates practical understanding of:

* Multiple Linear Regression
* Feature Engineering
* Correlation Analysis
* Multicollinearity
* Residual Analysis
* Regression Metrics
* Model Serialization
* Flask Deployment
* Machine Learning Project Structure

---

## Author

**Abhishek**

Machine Learning & Data Science Enthusiast
