import pandas as pd

# Series

# Create a Series
s = pd.Series([1, 3, 5, 7, 9])

# Return the first value of the Series
print(s[0])

# Add the labels to the data
s = pd.Series([1, 3, 5, 7, 9], index = ['a', 'b', 'c', 'd', 'e'])

# DataFrames

# Create a DataFrame
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

# Return the first row of the DataFrame
df = pd.loc[0]

# Load CSV into a DataFrame
df = pd.read_csv('data.csv')

# Return the entire DataFrame
df.to_string()

# Load JSON into a DataFrame
df = pd.read_json('data.json')

# Load data into a DataFrame
df = pd.DataFrame(data)

# Return the headers and the first 10 rows 
df.head(10)

# Return the first 5 rows
df.head()

# Return the last rows
df.tail()


# Empty Cells

# Remove rows with a NULL value
new_df = df.dropna()

# Replace NULL values with the number 130
new_df = df.fillna(130)

# Replace NULL values with the mean of the column
new_df = df.fillna(df.mean())

# Replace NULL values with the median of the column
new_df = df.fillna(df.median())

# Replace NULL values with the mode of the column
new_df = df.fillna(df.mode().iloc[0])


# Wrong Format

# Convert to a datetime format
df['Date'] = pd.to_datetime(df['Date'])


# Wrong Data

# Remove rows with wrong data
df.dropna(subset=['Date'])

# Replace wrong data
df.loc[7, 'Duration'] = 45


# Duplicates

# Discover duplicates
df.duplicated()

# Remove duplicates
df.drop_duplicates()


# Correlations

# Discover correlations
df.corr()

# Values are in the range -1 to 1
# 1 means strong positive correlation
# -1 means strong negative correlation
# 0 means no correlation
# A good correlation should be above 0.5


# Plotting

# plot() to create a line chart
df.plot()

# Scatter plot with "kind = scatter"
df.plot(kind='scatter', x='Duration', y='Calories')

# Histogram
df['Duration'].plot(kind='hist')




