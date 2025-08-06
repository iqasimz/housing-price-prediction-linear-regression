from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('data/train.csv')

# Step 2: Drop columns with more than 30% missing values
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.3].index
df = df.drop(columns=cols_to_drop)

# Step 3: Fill missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include='object').columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Step 4: One-Hot Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 5: Log-transform the target (SalePrice) to reduce skew
df_encoded['SalePrice'] = np.log1p(df_encoded['SalePrice'])

# Step 6: Split features and target
X = df_encoded.drop('SalePrice', axis=1)
y = df_encoded['SalePrice']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Print to confirm
print("✅ Data ready!")
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)

# Step 9: Initialize the model
model = LinearRegression()

# Step 10: Train the model
model.fit(X_train, y_train)

# Step 11: Predict on test set
y_pred = model.predict(X_test)

# Step 12: Evaluate performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 13: Print metrics
print("✅ Model Evaluation:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R² Score (Explained Variance): {r2:.3f}")

# Step 14: Plot predicted vs actual prices
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual Price (log scale)")
plt.ylabel("Predicted Price (log scale)")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 15: Interpret model coefficients
feature_names = X_train.columns
coefficients = model.coef_
# Create a DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
# Sort by absolute impact
coef_df['AbsCoeff'] = coef_df['Coefficient'].abs()
coef_df_sorted = coef_df.sort_values(by='AbsCoeff', ascending=False)
# Display top 10 most influential features
print("\n🔍 Top 10 Features Influencing House Price:")
print(coef_df_sorted[['Feature', 'Coefficient']].head(10))
plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df_sorted.head(10), x='Coefficient', y='Feature')
plt.title("Top 10 Most Influential Features")
plt.tight_layout()
plt.show()