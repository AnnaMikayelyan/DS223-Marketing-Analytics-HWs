# Churn Prediction and Retention Strategy - Survival Analysis

## Project Overview

This project aims to predict **customer churn** using **survival analysis** techniques, specifically applying the **Weibull model** and other parametric models to estimate the probability of customer retention over time. The objective is to understand the factors that influence churn risk, calculate the **Customer Lifetime Value (CLV)** for each customer, and determine the necessary **retention budget** to reduce churn.

## Dataset
The dataset contains the following columns:

- **ID**: Subscriber ID
- **region**: Region code
- **tenure**: Customer's lifetime (in months)
- **age**: Subscriber’s age
- **marital**: Subscriber’s marital status
- **address**: Number of years living at the same address
- **income**: Subscriber’s annual income (in thousands)
- **ed**: Education level
- **retire**: Retired status (Yes/No)
- **gender**: Gender of the subscriber
- **voice**: Whether the subscriber has voice service (Yes/No)
- **internet**: Whether the subscriber has internet service (Yes/No)
- **forward**: Whether call forwarding is active (Yes/No)
- **custcat**: Customer category (e.g., E-service, Plus service, Total service)
- **churn**: Whether the customer has churned (1: Churned, 0: Not Churned)

## Methodology

The analysis involves several **survival models** to predict customer churn:
1. **Weibull Distribution (AFT Model)**: Used to model customer retention time and churn risk.
2. **Other Parametric Models**: Including **Exponential**, **LogNormal**, **LogLogistic**, **GeneralizedGamma**, etc.
3. **Non-Parametric Models**: **Kaplan-Meier** and **Nelson-Aalen** methods were applied for comparison.
4. **Model Comparison**: Models were evaluated based on **AIC (Akaike Information Criterion)** and **Log-Likelihood**.



## How to Run the Project

1. **Clone this repository**:
  ```bash
  git clone https://github.com/AnnaMikayelyan/DS223-Marketing-Analytics-HWs.git
  ```
2. **Navigate to the project directory**:
  ```bash
  cd DS223-Marketing-Analytics-HWs/HW3
  ```
3.**Create and activate a virtual environment**:

3.1. **On macOS/Linux**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
3.2. **On Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
4. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
5. **Open and run cells in a Jupyter Notebook**:
  ```bash
  HW3_Survival_Analysis.ipynb
  ```

