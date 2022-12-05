# call functions and config modules
exec(open("./functions/functions.py").read())
exec(open("./config/config.py").read())

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# read in the credit risk data set
credit_risk= pd.read_csv("./data/credit_risk_data.csv")

# reset index to ID
credit_risk = credit_risk.set_index('SK_ID_CURR')

print('Null values in dataset are as follows:' '\n')
null = credit_risk.isnull().sum()
print(null[null>0])

# Create a boolean mask for categorical columns
categorical_mask = (credit_risk.dtypes == object)

# Get list of categorical column names
categorical_columns = credit_risk.columns[categorical_mask].tolist()
numeric_columns = credit_risk.columns[~categorical_mask].tolist()
colnames = list(credit_risk.columns.drop('TARGET'))
credit_risk[categorical_columns] = credit_risk[categorical_columns].astype(str)

names_to_encode = credit_risk.select_dtypes(include=[object], exclude=[float]).columns # create list of object names
credit_risk[names_to_encode] = MultiColumnLabelEncoder(columns = names_to_encode).fit_transform(credit_risk[names_to_encode].astype(str))

# # Convert df into a dictionary: df_dict
# df_dict = credit_risk.to_dict("records")
#
# # Create the DictVectorizer object: dv
# dv = DictVectorizer(sparse=False)
#
# # Apply dv on df: df_encoded
# df_encoded = dv.fit_transform(df_dict)
#
# df_encoded['TARGET'] = df_encoded['TARGET'].astype(str)
# X = credit_risk[colnames] # X value contains all the variables except labels
# y = credit_risk['TARGET'] # these are the labels

# create training test split data sets, with test size of 30% of the data
credit_risk['TARGET'] = credit_risk['TARGET'].astype(str)
X = credit_risk[colnames] # X value contains all the variables except labels
y = credit_risk['TARGET'] # these are the labe'

# create training test split data sets, with test size of 30% of the data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Instantiate the XGBClassifier: xg_cl
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
model.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = model.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# Create the DMatrix from X and y: churn_dmatrix
dmatrix = xgb.DMatrix(data=X, label=y, enable_categorical=True)

# Create the parameter dictionary: params
params = {"objective": "reg:logistic", "max_depth": 3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=dmatrix, params=params,
                    nfold=3, num_boost_round=5,
                    metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))

# full pipeline -----------------------------------------------------------------------------------------------------

credit_risk= pd.read_csv("./data/credit_risk_data.csv")
credit_risk = credit_risk.set_index('SK_ID_CURR')
credit_risk['TARGET'] = credit_risk['TARGET'].astype(str)
X = credit_risk[colnames] # X value contains all the variables except labels
y = credit_risk['TARGET'] # these are the labe'

# create training test split data sets, with test size of 30% of the data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer()),
         ('scaler', StandardScaler(with_mean=False)),
         ("xgb_model", xgb.XGBClassifier())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)
cross_val_score(xgb_pipeline,X.to_dict("records"), y, scoring="roc_auc", cv=2)
# Cross-validate the model
# roc_auc
cross_val_scores = cross_val_score(xgb_pipeline,X.to_dict("records"), y, scoring="roc_auc", cv=10)


# Print the 10-fold RMSE
print("10-fold ROC mean: ", np.mean((cross_val_scores)))