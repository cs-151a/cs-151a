# CSE-151A Group Project - Milestone 5: Final Report

Link to Jupyter Notebook: [https://github.com/cs-151a/cs-151a/blob/Milestone4/stocks.ipynb](https://github.com/cs-151a/cs-151a/blob/Milestone4/stocks.ipynb)

## Introduction

For our project, we chose the S&P 500 as our dataset. The reason being is because it is one of the most important market benchmarks representing the largest US companies. We knew that it had extensive data available making it the ideal dataset to train models. The data is high quality and reliable compared to regular stocks which may contain gaps in the data or irregularities. In addition, the index is less volatile than individual stocks, making it easier to model. 

Now you may ask why this is cool. It is because it can help us track money and how we can add more money to our pockets. It is highly liquid and widely traded making any insights we receive highly applicable. On top of that by picking this dataset we get the “wisdom of the crowd” effect. Instead of focusing on a single company, we can capture the collective performance of the strongest corporate entities in the United States. These companies represent about 80% of the available market capitalization in the U.S. stock market. 

There is strong real-world impact and accessibility. Most people’s retirement is based on the 401ks that they receive which are heavily based on the S&P 500 funds. It’s the most common fund for benchmark performance. It represents rich data properties. The data can show interesting patterns both within the long term and the short term. It captures broader economic cycles and market sentiment, contains both rational price movements and emotional/behavioral aspects and has survived major historical events (crashes, recoveries, tech bubbles, financial crises). In essence, we can say that it is cool because we are not just modeling stocks. We are looking at the measure of American economic health that millions of people rely on for their economic well being. 

The models we chose for our project were the random forest regression model and the XGBoost model. When it comes to the random forest regression model, it serves as a good baseline model with reliable results. It handles non-linear relationships between the features and stock movements. Furthermore, it provides feature importance scores and also allows us to easily visualize the contribution of our various features. Finally, this model is relatively efficient computationally which was an added bonus.

Our second model was the XGBoost model. This was our more in depth model as it can capture non-linear relationships that the Linear Regression model might miss. It does a great job in handling complex interactions between features while generally providing better predictive performances compared to the Linear Regression model. It contains built-in feature importance metrics to understand what drives predictions. Due to these abilities, it is resistant to overfitting through regularization. 

These models help us go further with the S&P 500. Having a good predictive model has a broader impact that is quite significant. Using good models like ours, investors can better understand the risks of the market and identify new opportunities. Portfolio managers can improve their strategies with hedge funds. It can help the regular person comprehend which factors most influence market movements. In addition, it can be used to develop new trading strategies as well as risk management systems. In reality, it can help us innovate our trading strategies and help provide insights into the overall market behavior and the economic health of the United States of America.

## Methods
(All relevant code can be found in our notebook, linked above)

### Data Exploration

The dataset consisted of 602,962 rows and 9 columns, which included the following features:
- **Data** (object)
- **Open** (float64)
- **High** (float64)
- **Low** (float64)
- **Close** (float64)
- **Volume** (int64)
- **Dividends** (float64)
- **Stock Splits** (float64)
- **Company** (object)

The first five rows of the dataset were previewed for context. Additionally, a statistical summary of the numerical columns was generated, which highlighted significant variability in key financial metrics.
## INSERT IMAGE HERE

The dataset contained 491 unique companies, each identified by a ticker symbol (e.g., AAPL, MSFT, GOOGL, AMZN). Missing value checks confirmed that the dataset was complete with no missing values.

Several visualization techniques were employed to understand data distribution and relationships (figures for each are available in the “Results” section):

- A correlation heatmap was generated to identify linear relationships between numeric columns
- A pair plot was used to visualize relationships between features
- Distribution plots were created for individual numeric columns to analyze the spread and potential skewness
  
The exploration step provided insights into the dataset's structure and statistical patterns, setting the stage for further preprocessing and modeling

### Preprocessing

We took the following steps to preprocess our data for further modeling:

- **Dropped null and duplicate rows**
  - Any rows containing null values were removed to ensure data completeness
  - Any duplicate rows were dropped to eliminate redundant data

- **Removed irrelevant columns**
  - The "Dividends" and "Stock Splits" columns were dropped as they did not contribute to the analysis

- **Date conversion**
  - The "Date" column was converted from strings to datetime objects using Pandas’ to_datetime function with UTC specified. This allowed for better date-based calculations and sorting

- **Filtered out companies with insufficient data**
  - Stocks with less than three years of observations (measured between 2018 and 2024) were removed. Companies were filtered by checking for at least three consecutive years of data in the "Date" column

- **Feature scaling**
  - The numerical columns ("Open," "High," "Low," "Close," "Volume") were scaled using Min-Max Scaling to normalize values between 0 and 1. This ensured that all features contributed equally to the model

Next, we had to calculate financial metrics, such as returns and volatility, for each stock.

- **Daily Returns** (time-series representation of how each stock’s price changed each day)
  - Calculated as the percentage change in the “Close” column prices
  - Grouped by company

- **Annualized Return** (average annual performance of an investment)
  - Calculated by taking the mean of daily returns for each company and scaling it by 252 (approximate amount of trading days per year)

- **Annualized Volatility** (how much a stock price varies over a year)
  - Calculated by taking the standard deviation of daily returns, and scaling by 252

- **Risk-Adjusted Return** (measures profit of an investment without considering risk)
  - Calculated by dividing the annualized return by the annualized volatility

Finally, we used each of these financial metrics to calculate Sharpe Ratio for the dataset, a more comprehensive risk-adjusted metric:

- **Sharpe Ratio** (compares the return of an investment with its risk)
  - Calculated by dividing the annualized excess return by the annualized volatility

Ultimately, the preprocessing steps prepared the dataset by cleaning, scaling, and enriching it with essential financial metrics, such as daily returns, annualized returns, volatility, and Sharpe ratios. These transformations ensured that the data was both accurate and suitable for subsequent machine learning models.

### Model 1

#### Model Overview

The first model implemented was a Random Forest Regressor, chosen for its ability to handle non-linear relationships and provide feature importance scores.

#### Model Architecture

The Random Forest Regressor was configured with the following hyperparameters:

- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
  
#### Data Preparation

- **Feature Selection**
  - The following weighted features were generated based on their importance to the target prediction:
    - Weighted_Risk_Adjusted_Return: 0.5
    - Weighted_Sharpe: 0.3
    - Weighted_Volatility: 0.2
    
- **Target Variable**
  - The target variable, Future_Return, was calculated as the mean future return over a 30-day period
  - Missing values in the target variable were filled using the median
  
- **Feature Scaling**
  - Features were scaled using a RobustScaler to handle outliers and ensure numerical stability
  
#### Training and Testing

- The dataset was split into training and testing sets using an 80:20 ratio with train_test_split
- The Random Forest Regressor was trained on the scaled features from the training set, and predictions were made on the testing set

### Model 2

#### Model Overview

The second model implemented was an XGBoost Regressor. This ensemble learning method was chosen for its efficiency in handling complex datasets and its ability to model non-linear relationships effectively.

#### Model Architecture

The XGBoost model was trained with the following hyperparameters:
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 6
- min_child_weight: 1
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 0
  
These parameters were initially set before hyperparameter tuning, which is discussed later.

#### Data Preparation

- **Data Splitting**
  - The dataset was split into training (80%) and test (20%) subsets
    
- **Feature Scaling**
  - The data was scaled using the RobustScaler to mitigate the impact of outliers
    
#### Training and Testing

The model was trained using the following approach:

- **Cross-Validation**
  - Five-fold cross-validation was performed to assess the stability and robustness of the model
 
- **Model Fit**
  - The model was trained on the scaled training data and evaluated on the test set
    
#### Tuning and Hyperparameter Optimization

To optimize the model’s performance, a Grid Search approach was implemented. This method was used to find the optimal set of hyperparameters for the model, which will be discussed in the “Results” section for Model 2.

