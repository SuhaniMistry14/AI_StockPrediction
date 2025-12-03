# pip install pandas numpy matplotlib seaborn scikit-learn tensorflow streamlit

# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt


# LOADING THE DATASET 
df = pd.read_csv("data/tesla.csv")

# ROWS 
# print(df.head())

# INFO
# print(df.info())

# DESCRIBE 
# print(df.describe())

# CLEANING 
# COVERTING DATE INTO DATETIME
df ['Date'] = pd.to_datetime(df['Date'])
# SORTING 
df = df.sort_values('Date')

# CHECKING FOR MISSING VALUES
print("Missing Values : ", df.isnull().sum())   
# No missing values found

# Analysing Data
# Plotting  the closing prices by dates
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.title("Closing Price By Dates")
plt.xlabel("Dates")
plt.ylabel("Closing Price")
plt.show()


# Feature Engineering 
# Adding Moving Averages 
# Short Term - Move50
# Long Term - Move200
df['Move50'] = df['Close'].rolling(50).mean()
df['Move200'] = df['Close'].rolling(200).mean()

# Plotting closing price v/s dates using moving averages
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.plot(df['Date'], df['Move50'], label='50 Days Moving Averages', color='orange')
plt.plot(df['Date'], df['Move200'], label = '200 Days Moving Average ', color='green')
plt.title("Closing Prices with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.show()
