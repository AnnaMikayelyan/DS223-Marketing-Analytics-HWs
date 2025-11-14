# Churn Prediction and Retention Strategy - Survival Analysis

## Project Overview

This project aims to predict **customer churn** using **survival analysis** techniques, specifically applying the **Weibull model** and other parametric models to estimate the probability of customer retention over time. The objective is to understand the factors that influence churn risk, calculate the **Customer Lifetime Value (CLV)** for each customer, and determine the necessary **retention budget** to reduce churn.

## Dataset Description

The dataset contains **customer demographic and usage data**, with the target variable being **churn** (whether a customer has churned, represented as 1 for "Yes" and 0 for "No"). Key features include:
- **Tenure**: Duration of customer subscription (in months).
- **Age**: Customer's age.
- **Marital Status**: Whether the customer is married or unmarried.
- **Income**: Annual income of the customer (in thousands).
- **Education**: Customer's education level.
- **Retirement Status**: Whether the customer is retired.
- **Gender**: Gender of the customer.
- **Internet and Voice Access**: Whether the customer has internet or voice services.
- **Region**: Geographic region of the customer.
- **Churn**: Target variable indicating whether the customer churned.

## Methodology

The analysis involves several **survival models** to predict customer churn:
1. **Weibull Distribution (AFT Model)**: Used to model customer retention time and churn risk.
2. **Other Parametric Models**: Including **Exponential**, **LogNormal**, **LogLogistic**, **GeneralizedGamma**, etc.
3. **Non-Parametric Models**: **Kaplan-Meier** and **Nelson-Aalen** methods were applied for comparison.
4. **Model Comparison**: Models were evaluated based on **AIC (Akaike Information Criterion)** and **Log-Likelihood**.

### Steps Taken:
- **Data Preprocessing**: Converting categorical variables to numerical codes, applying **Label Encoding** for binary features, and **One-Hot Encoding** for multi-category features.
- **Model Fitting**: Fitting different survival models to estimate churn probabilities and survival functions.
- **Retention Budget Calculation**: Using **CLV** and survival probabilities to calculate the annual retention budget required to retain at-risk customers.

## Key Findings

- **Churn Probability**: The **Weibull model** predicted that **7.93% of customers** are likely to churn within the next 12 months.
- **Customer Lifetime Value (CLV)**: The **average CLV** per customer is approximately **$2000**.
- **Retention Budget**: To retain at-risk customers (those likely to churn), the estimated **annual retention budget** is around **$79,000**, based on **50% of the CLV** allocated for retention efforts.
- **Most Valuable Segments**:
  - **Retired customers** show higher **CLV** and lower churn rates, making them valuable segments.
  - Customers with **no internet access** also exhibit lower churn rates and higher retention.

## Retention Strategy

Based on the findings, here are some retention strategies:
1. **Targeted Retention Campaigns**: Focus on customers with **low tenure** or who are likely to churn soon, especially those without internet access or voice services.
2. **Loyalty Programs**: Offer incentives to **retired customers** and those with **high CLV** to retain them longer.
3. **Customer Engagement**: Implement proactive communication, loyalty rewards, and personalized offers to increase customer satisfaction and reduce churn.

## How to Run the Project

### Dependencies:
To run the analysis, you’ll need to install the following libraries:

```bash
pip install pandas numpy lifelines scikit-learn matplotlib

