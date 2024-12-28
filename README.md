# bulldozer-sale-price
pip install pandas numpy scikit-learn matplotlib seaborn
import pandas as pd

# Load the dataset
df = pd.read_csv('Train.csv', low_memory=False)

# Display the first few rows of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Fill missing values for 'YearMade' with the median
df['YearMade'].fillna(df['YearMade'].median(), inplace=True)

# Drop columns with too many missing values or irrelevant columns
df.drop(['SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'url', 'saledate'], axis=1, inplace=True)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Check the shape of the dataset after preprocessing
print(df.shape)
# Check for missing values
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Fill missing values for 'YearMade' with the median
df['YearMade'].fillna(df['YearMade'].median(), inplace=True)

# Drop columns with too many missing values or irrelevant columns
df.drop(['SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'url', 'saledate'], axis=1, inplace=True)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Check the shape of the dataset after preprocessing
print(df.shape)
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of sale prices
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of sale prices
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()
from sklearn.model_selection import train_test_split

# Define features and target variable
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances')
plt.show()
