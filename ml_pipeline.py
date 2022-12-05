# call functions and config modules
exec(open("./functions/functions.py").read())
exec(open("./config/config.py").read())


# Basics
import pandas as pd
import xgboost as xgb
# Pipeline
from sklearn.pipeline import Pipeline
# Scaler for standardization
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from pylab import rcParams
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

# read in the credit risk data set
credit_risk= pd.read_csv("./data/credit_risk_data.csv")
# reset index to ID
credit_risk = credit_risk.set_index('SK_ID_CURR')
credit_risk = credit_risk.dropna()
categorical_mask = (credit_risk.dtypes == object)
# Get list of categorical column names
categorical_columns = credit_risk.columns[categorical_mask].tolist()
numeric_columns = credit_risk.columns[~categorical_mask].tolist()
credit_risk[categorical_columns] = credit_risk[categorical_columns].astype(str)

X = credit_risk.drop('TARGET', axis=1)
y = credit_risk.TARGET
# Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Tasks to perform
cat_steps = [('OHE', OneHotEncoder(drop='first',
                                   handle_unknown='error'))]
# Pipeline object
pipe_cat = Pipeline(cat_steps)
# Tasks to perform
num_steps = [('scale', StandardScaler()) ]
# Pipeline object
pipe_num = Pipeline(num_steps)

# Extracting the names of the categorical variables
categorical_vars = X.select_dtypes('object').columns.tolist()
# Extracting the names of the numerical variables
numerical_vars = X.select_dtypes('number').columns.tolist()
# Creating the multilayer pipe
one_pipe = ColumnTransformer(transformers=[
          ('numbers', pipe_num, numerical_vars),
          ('categories', pipe_cat, categorical_vars) ] )

# Final Pipeline
modeling = Pipeline([('preprocess', one_pipe),
                     ('model', xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123))
                     ])
# Fit
modeling.fit(X_train, y_train)

print("The accuracy of XGBoost model is:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
print("The precision score is: ", precision_score(y_true=y_test, y_pred=preds,  average="binary"))
print("The recall score is: ", recall_score(y_true=y_test, y_pred=preds,  average="binary"))








class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value

    def fit(self, X, y=None):
        if self.method == 'mean':
            self.value = X[self.features].mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max() - self.min
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features] - self.min) / self.range
        return X_transformed


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True),
                                   pd.DataFrame(self.encoder.transform(X[self.features]),
                                                columns=self.encoder.get_feature_names(self.features))],
                                  axis=1)
        return X_transformed


def calculate_roc_auc(model_pipe, X, y):
    """Calculate roc auc score.

    Parameters:
    ===========
    model_pipe: sklearn model or pipeline
    X: features
    y: true target
    """
    y_proba = model_pipe.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_proba)

credit_risk= pd.read_csv("./data/credit_risk_data.csv")
credit_risk = credit_risk.set_index('SK_ID_CURR')
SEED = 42
TARGET = 'TARGET'
FEATURES = credit_risk.columns.drop(TARGET)

NUMERICAL = credit_risk[FEATURES].select_dtypes('number').columns
CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))


pipe = imbpipeline([
    ('smote', SMOTE(random_state=11)),
    ('num_imputer', Imputer(NUMERICAL, method='mean')),
    ('scaler', Scaler(NUMERICAL)),
    ('cat_imputer', Imputer(CATEGORICAL)),
    ('encoder', Encoder(CATEGORICAL)),
    ('model', LogisticRegression())#xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123))
])



preds = pipe.predict(X_test)
