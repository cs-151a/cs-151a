# CSE-151A

Link to Jupyter Notebook: [https://github.com/cs-151a/cs-151a/blob/main/stocks.ipynb](https://github.com/cs-151a/cs-151a/blob/main/stocks.ipynb)

### Dataset Preprocessing:
**1. Change format of 'Date' column**
* The current 'Date' column is in object format (ex. 2018-11-29). This is not ideal for the time-series analysis we wish to perform on the data. So, we will convert to a datetime format, allowing for more convenient month and year extraction as well as time-based indexing.

**2. Sort each company's data by Date**
* To assist in the time-series analysis mentioned in Step 1, we will put each company's data in sequential order based on 'Date' data.

**3. Handle duplicate data**
* Since duplicates are unnecessary and could potentially harm the effectiveness of our model, we will remove any duplicate rows for the same 'Date' data.

**4. Encode categorical data**
* We cannot build our model on non-numerical data values, so will need to encode any such instances in our dataset. Particularly, the 'Company' column contains strings for each company's stock symbol, so we will use one-hot encoding to convert them to numerical data.

**5. Drop unnecessary columns**
* We can drop the 'Dividends' and 'Stock Splits' columns because these data are not necessary in building our model. Dropping them will reduce potential confusion and better streamline the process of training our model.
