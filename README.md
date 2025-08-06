--A Linear Regression project to predict house prices using the Ames Housing Dataset.

--Objectives:
	•	Predict housing prices using all the features in the dataset
	•	Perform end-to-end data cleaning and feature encoding
	•	Apply a log transformation to normalize target
	•	Train a Linear Regression model
	•	Evaluate using MAE, RMSE, and R²
	•	Interpret model coefficients to explain predictions
 
--Methodology:
	•	Dropped columns with >30% missing values
	•	Filled missing numeric columns with median, categorical with mode
	•	One-hot encoded categorical variables
	•	Applied log transform to target (SalePrice) for better linear modeling
	•	Used LinearRegression() from Scikit-Learn
	•	Trained on 80% of data, evaluated on 20%
 
--Feature Coefficients:
	•	Positive → raises price
	•	Negative → reduces price
