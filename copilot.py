import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
#Label encodings
from sklearn.preprocessing import LabelEncoder 
#Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier

#Read the data
fileName = "train.csv"
df = pd.read_csv(fileName)
#print(df.head())

######################################
# Data Preprocessing
######################################
#Check for missing values
missing_values = df.isnull().sum()
#print(missing_values)
#print(len(df)," rows before dropping missing values")
df = df.dropna()
#print(len(df), " rows after dropping missing values")

#drop columns that are not needed
df.drop(['Name','PassengerId'], axis=1, inplace=True)

# Create new columns from 'Cabin', as this column contains three information (deck, number and side)
df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
df['Cabin_Num'] = df['Cabin_Num'].astype(int)
df.drop('Cabin', axis=1, inplace=True)

# Check to see the number of unique values in 'Destination'
#unique_destinations = df['Destination'].unique()
#print("Unique values in 'Destination':", unique_destinations)

# Label encode the 'HomePlanet', 'Cabin_Deck', and 'Cabin_Side' columns
# they were originally categorical columns
label_encoder = LabelEncoder()
df['HomePlanet'] = label_encoder.fit_transform(df['HomePlanet'])
df['Cabin_Deck'] = label_encoder.fit_transform(df['Cabin_Deck'])
df['Cabin_Side'] = label_encoder.fit_transform(df['Cabin_Side'])
df['Destination'] = label_encoder.fit_transform(df['Destination'])

# Convert 'VIP' and 'CryoSleep' columns to 1 and 0
df['VIP'] = df['VIP'].astype(str).str.strip()
df['VIP'] = df['VIP'].fillna('False').map({"True": 1, "False": 0}).astype(int)

df['CryoSleep'] = df['CryoSleep'].astype(str).str.strip()
df['CryoSleep'] = df['CryoSleep'].map({'True': 1, 'False': 0}).astype(int)


# Visualize the regression coefficient between all features and 'Transported'
"""correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()"""

######################################
# Training Linear Regression Model
######################################

# Split the data into features and target variable
X = df.drop('Transported', axis=1)
y = df['Transported']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = np.mean(np.round(y_pred) == y_test)

print(f"Mean Squared Error: Linear Regression {mse}")
print(f"R-squared: Linear Regression {r2}")
print(f"Accuracy: Linear Regression {accuracy}")

######################################
# Training Neural Network Model 
######################################

# Define the neural network model
nn_model = Sequential()
nn_model.add(tf.keras.Input(shape=(X_train.shape[1],)))
nn_model.add(Dense(7, activation='relu'))
nn_model.add(Dense(7, activation='relu'))
nn_model.add(Dense(3, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# Compile the model
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs = 100, batch_size = 10000, verbose = 0, validation_split = 0.2)

# Make predictions
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
accuracy = np.mean(y_pred == y_test)

print(f"Mean Squared Error: Neural Network {mse}")
print(f"R-squared: Neural Network {r2}")
print(f"Accuracy: Neural Network {accuracy}")

######################################
# Training Random Forest Model
######################################

# Initialize the random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
accuracy_rf = np.mean(y_pred_rf == y_test)

print(f"Mean Squared Error: Random Forest {mse_rf}")
print(f"R-squared: Random Forest {r2_rf}")
print(f"Accuracy: Random Forest {accuracy_rf}")
