## Project Overview

This repository contains the implementation of **Assignment 1 (DS223: Marketing Analytics)**, focusing on the **Bass Diffusion Model** for innovation adoption. The project applies the Bass model to estimate and forecast the diffusion of **A Slimmer Hiking Shoe** from TIME’s Best Inventions 2024, using **Merrell Moab Hiking Shoe Series** as a comparable past innovation.


## Objectives

1. Identify a modern innovation and its historical analog.

2. Collect time-series data representing market adoption.

3. Estimate Bass model parameters (p, q, M) using the historical data.

4. Predict diffusion for the new innovation based on fitted parameters.

5. Interpret results and discuss adoption behavior trends.


## Data Sources

Two datasets were used in the scope of the projects about the U.S. athletic footwear market, which reasonably include hiking footwear trends:

- **Revenue** of the athletic footwear segment in the United States (2018–2030), in billion USD.  
  Available at: <https://www.statista.com/forecasts/246496/athletic-footwear-industy-revenue>  
- **Volume** of the athletic footwear segment in the United States (2018–2030), in millions of pairs.  
  Available at: <https://www.statista.com/forecasts/1381132/athletic-footwear-industy-volume>


## Images Directory Description

* **header_image.png** : Header and product comparison image showing Merrell SpeedARC Surge BOA and Merrell Moab Ventilator, illustrating the evolution from traditional to smart footwear.

* **Bass_Model_Fit_Comulative_Revenue_and_Volume.png** – Bass model fit plots comparing actual vs. predicted cumulative revenue and cumulative volume, visualizing model accuracy.

* **Bass_Model_Fit_Revenue_and_Volume.png** – Bass model fit plots for annual revenue and annual volume, showing observed and predicted values over time.

* **Bass_Model_Prediciton.png** – Combined diffusion forecast showing new adopters per year and cumulative adoption, representing the predicted innovation diffusion trend.

* **New_Adopters.png** – Bar chart illustrating the estimated number of new adopters per year for the Merrell SpeedARC Surge BOA, highlighting gradual adoption decline over time.


## Methodology

The project uses the Bass Diffusion Model, defined by the parameters:
* p (innovation coefficient) which captures influence of external communication.

* q (imitation coefficient) which captures internal influence or word-of-mouth.

* M (market potential) which represents the maximum number of adopters.

Parameter estimation was done via non-linear least squares fitting using scipy.optimize.curve_fit.
Both annual and cumulative adoption models were fitted to compare behavioral trends. Forecasts were generated for 2024–2034, predicting the adoption curve of the SpeedARC Surge BOA based on the Moab’s market behavior.

## Setup Instructions

This section provides the necessary steps to reproduce the analysis and results presented in this project.  
To ensure full functionality, follow the setup process below before running the notebook.  

1. **Clone the repository**  
   Open your terminal and clone the repository to your local system:  
   ```bash
   git clone [https://github.com/AnnaMikayelyan/DS223-Bass-Model.git]
   
   cd <your-repo-name>
   ```

2. **Create a virtual environment**  
   It is recommended to use a virtual environment to isolate dependencies.  
   ```bash
   python -m venv env
   source env/bin/activate    # For macOS/Linux
   env\Scripts\activate       # For Windows
   ```

3. **Install dependencies**  
   Install all required Python libraries listed in the `requirements.txt` file:  
   ```bash
   pip install -r requirements.txt
   ```
   The key libraries include:  
   - `pandas` – data manipulation and preprocessing  
   - `numpy` – numerical computations  
   - `matplotlib` and `seaborn` – data visualization  
   - `scipy` – Bass Model parameter estimation (`curve_fit`)  
   - `openpyxl` – reading Excel files  

4. **Run the Jupyter Notebook**  
   Start Jupyter Notebook in your working directory:  
   ```bash
   jupyter notebook
   ```
   Then open and run all cells in the file:  
   ```
   DS 223 Homework 1.ipynb
   ```

## References

1. **TIME** (2024). *A Slimmer Hiking Shoe — Merrell SpeedARC Surge BOA.* TIME’s Best Inventions 2024.  
   Retrieved from [https://time.com/collection/best-inventions-2024/](https://time.com/collection/best-inventions-2024/)

2. **ISPO** (2024). *ISPO Award Winner: Merrell SpeedARC Surge BOA.*  
   Retrieved from [https://www.ispo.com/en/promotion/ispo-award-winner-merrell-speedarc-surge-boa](https://www.ispo.com/en/promotion/ispo-award-winner-merrell-speedarc-surge-boa)

3. **Merrell** (2021). *Merrell Debuts Extensions of Its Most Popular Hiking Boot Franchise with the Moab Speed and Moab Flight.* PR Newswire.  
   Retrieved from [https://www.prnewswire.com/news-releases/merrell-debuts-extensions-of-its-most-popular-hiking-boot-franchise-with-the-moab-speed-and-moab-flight-301254309.html](https://www.prnewswire.com/news-releases/merrell-debuts-extensions-of-its-most-popular-hiking-boot-franchise-with-the-moab-speed-and-moab-flight-301254309.html)

4. **GearPatrol** (2024). *Merrell Moab 3 Waterproof Camo Review.*  
   Retrieved from [https://www.gearpatrol.com/outdoors/introducing-merrell-moab-3-waterproof-camo/](https://www.gearpatrol.com/outdoors/introducing-merrell-moab-3-waterproof-camo/)

5. **Statista** (2024a). *Revenue in the Athletic Footwear Segment in the United States from 2018 to 2030 (in billion U.S. dollars).*  
   Retrieved from [https://www.statista.com/forecasts/246496/athletic-footwear-industy-revenue](https://www.statista.com/forecasts/246496/athletic-footwear-industy-revenue)

6. **Statista** (2024b). *Volume in the Athletic Footwear Segment in the United States from 2018 to 2030 (in millions).*  
   Retrieved from [https://www.statista.com/forecasts/1381132/athletic-footwear-industy-volume](https://www.statista.com/forecasts/1381132/athletic-footwear-industy-volume)

7. **Course Slides** (2024). *DS-223: Bass Model.* [PDF file]. American University of Armenia.
