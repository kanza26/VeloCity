import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle



df=pd.read_csv(r'C:\Users\Mehveen Fasih\Downloads\chicago-traffic-tracker-historical-congestion-estimates-by-segment-2018-current-1 (2).csv')
print(df)
df.info()
df.isnull().sum()
# Corrected drop command with exact column names
df.drop(['STREET','SPEED', 'LENGTH', 'SEGMENT_ID', 'DIRECTION', 'STREET_HEADING', 
         'COMMENTS', 'BuselessCOUNT', 'MESSAGE_COUNT', 'HOUR', 'DAY_OF_WEEK', 
         'MONTH', 'RECORD_ID', 'START_LOCATION', 'END_LOCATION', 
         'Community Areas', 'Zip Codes', 'Wards'], axis=1, inplace=True)

#df.drop(['cOMMENTS'], axis=1, inplace=True)
print(df)
#Save the cleaned DataFrame to a CSV file
df.to_csv('Traffic_cleaned_data.csv', index=False)  # Set index=False to avoid adding row numbers

# Save the trained model using pickle

plt.figure(figsize=(15, 6))

street_pair_bus_counts = df.groupby(['FROM_STREET', 'TO_STREET'])['BUS_COUNT'].sum().reset_index()

# Plotting
plt.figure(figsize=(140, 100))
sns.barplot(data=street_pair_bus_counts, x='FROM_STREET', y='BUS_COUNT', hue='TO_STREET')
plt.xticks(rotation=90)
plt.title("Total Bus Count by Street Pairs (FROM_STREET to TO_STREET)")
plt.xlabel("From Street")
plt.ylabel("Bus Count")
plt.legend(title="To Street", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')

# Check for any conversion issues
print(df['TIME'].isnull().sum(), "rows had issues in conversion.")

# Now, group by TIME to get the sum of BUS_COUNT at each date and time
bus_count_by_time = df.groupby('TIME')['BUS_COUNT'].sum().reset_index()

# Display the first few rows to check the data
print(bus_count_by_time.head())

# Plotting the bus count over time
plt.figure(figsize=(14, 8))
sns.lineplot(data=bus_count_by_time, x='TIME', y='BUS_COUNT')
plt.title("Bus Count Over Time")
plt.xlabel("Date and Time")
plt.ylabel("Bus Count")
plt.xticks(rotation=45)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.show()
#Explanation:plt.show()
df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
bus_count_by_time_street = df.groupby(['TIME', 'FROM_STREET', 'TO_STREET'])['BUS_COUNT'].sum().reset_index()

# Display the first few rows to check the data
print(bus_count_by_time_street.head())

# Plotting the bus count over time for each street pair
plt.figure(figsize=(14, 8))
sns.lineplot(data=bus_count_by_time_street, x='TIME', y='BUS_COUNT', hue='FROM_STREET', style='TO_STREET', markers=True, dashes=False)
plt.title("Bus Count Over Time by Street Pair")
plt.xlabel("Date and Time")
plt.ylabel("Bus Count")
plt.xticks(rotation=45)

# Format x-axis to show both date and time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.legend(title="From Street to Street", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

# Convert TIME to datetime
df['TIME'] = pd.to_datetime(df['TIME'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Extract additional features from TIME
df['HOUR'] = df['TIME'].dt.hour
df['DAY_OF_WEEK'] = df['TIME'].dt.dayofweek
df['MONTH'] = df['TIME'].dt.month


X = df[['FROM_STREET', 'TO_STREET', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE']]

y = df['BUS_COUNT']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['FROM_STREET', 'TO_STREET']),
         ('num', StandardScaler(), [ 'HOUR', 'DAY_OF_WEEK', 'MONTH', 
                                'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE'])

    ]
)

# Create a pipeline to transform data
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply transformations and split data
X_transformed = pipeline.fit_transform(X)
print(X_transformed)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)






#
from lightgbm import LGBMRegressor

# Adjusted model parameters
model = LGBMRegressor(n_estimators=3000, max_depth=10, learning_rate=0.1, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

maegb = mean_absolute_error(y_test, y_pred)
msegb = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error: gb", maegb)
print("Mean Squared Error gb:", msegb)
print("R-squared:", r2)
import gc  # For garbage collection

try:
#    del 
    del X_train, X_test, y_train, y_test  # Only if you no longer need them for further tasks
except NameError:
    pass  # If they were already deleted or not defined, this will ignore the error

# Force garbage collection to release memory
gc.collect()

# Verify if the memory is cleared by running a small statement, like checking memory usage

print("Unnecessary models and variables cleared, only lighgbm model retained.")
import pandas as pd
from datetime import datetime

def predict_bus_count(new_data, model, pipeline):
    
    # Ensure TIME is in datetime format
    new_data['TIME'] = pd.to_datetime(new_data['TIME'], errors='coerce')
   
    new_data['HOUR'] = new_data['TIME'].dt.hour
    new_data['DAY_OF_WEEK'] = new_data['TIME'].dt.dayofweek
    new_data['MONTH'] = new_data['TIME'].dt.month
    X_new = new_data[['FROM_STREET', 'TO_STREET', 'HOUR', 'DAY_OF_WEEK', 
                      'MONTH', 'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE']]
    
    # Apply transformations with the pipeline
    X_new_transformed = pipeline.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_transformed).astype(int)

    
    return predictions


new_data = pd.DataFrame({
    'TIME': ['11/20/2024 12:31:00 '],
    'FROM_STREET': ['Michigan'],
    'TO_STREET': ['Michigan'],
    'START_LATITUDE': [41.89],
    'START_LONGITUDE': [-87.62],
    'END_LATITUDE': [41.89],
    'END_LONGITUDE': [-87.62],
    
})

predictions = predict_bus_count(new_data, model, pipeline)
print("Predicted bus count:", predictions)
with open('traffic_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    print('model saved')


