# CSE-151A

Link to Jupyter Notebook: [https://github.com/cs-151a/cs-151a/blob/main/stocks.ipynb](https://github.com/cs-151a/cs-151a/blob/main/stocks.ipynb)

### Dataset Preprocessing:
**1. Change format of 'Date' column**
* The current 'Date' column is in object format (ex. 2018-11-29 00:00:00-05:00). This is not ideal for the time-series analysis we wish to perform on the data. So, we will convert to a datetime format, allowing for more convenient month and year extraction as well as time-based indexing. Additionally, we will drop the time from this column entirely, as we simply need the date and year for each company's stock entries.

**2. Sort each company's data by Date**
* To assist in the time-series analysis mentioned in Step 1, we will put each company's data in sequential order based on 'Date' data. Since we have 5 years worth of data, as well as data for each day the market is open, sorting each company's data sequentially will be crucial in keeping our dataset organized and efficiently readable.

**3. Handle duplicate data**
* If for some reason a company has multiple data entries for the same date in the 'Date' column, we will remove any extraneous entries because they are not necessary for training the model. On top of that, the model can become confused if it encounters more than or less than 491 entries for the same date, so we need to make sure each date contains solely 491 entries as we have exactly 491 ticker symbols.

**4. Encode categorical data**
* We cannot build our model on non-numerical data values, so we will need to encode any such instances in our dataset. Particularly, the 'Company' column contains strings for each company's stock symbol, so we will use one-hot encoding to convert them to numerical data. For instance, consider the company symbols 'AAPL', 'MSFT', and 'TSLA'. By one-hot encoding these values, we would make a row with three new columns similar to this: ('Company_AAPL', 'Company_MSFT', 'Company_TSLA'). However, they will be filled with zeros, and only a singular one to represent a specific company. For instance, AAPL would be (1, 0, 0) whereas TSLA would be (0, 0, 1) in our sample case. 

**5. Drop unnecessary columns**
* We can drop the 'Dividends' and 'Stock Splits' columns because these data are not necessary in building our model. This is due to the fact that our model will be focused on calculating predicted risk-adjusted return, sharpe ratios, as well as volatility, none of which require dividens or stock splits to calculate. Not only that, but it's not fair to consider these data columns in training our model because there exist companies that don't partake in dividends or stock splits despite generating lots of returns. Thus, dropping them will reduce potential confusion and better streamline the process of training our model. 

**6. Predicted Risk-Adjusted Return**
* Calculates and ranks the performance of stocks based on their risk-adjusted return. It begins by converting the Date column to a datetime format and sorting the data by Company and Date. Using grouped calculations for each company, it computes daily returns, annualized return (mean of daily returns scaled to a year), annualized volatility (standard deviation of daily returns scaled to a year), and the risk-adjusted return (annualized return divided by annualized volatility). These metrics are compiled into a results DataFrame. The results are then ranked by Risk-Adjusted Return in descending order, with a Rank column added for clarity. The output shows which stocks offer the best return relative to risk.