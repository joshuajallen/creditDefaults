# import librariers for machine learning
from statsmodels.formula.api import ols
import pycaret
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from pylab import rcParams
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_predict
from pycaret.classification import *


# read in the credit risk data set
credit_risk= pd.read_csv("./data/credit_risk_data.csv")

# reset index to ID
credit_risk = credit_risk.set_index('SK_ID_CURR')

colnames = list(credit_risk.columns.drop('TARGET')) # create list of predictor names
categorical_names = credit_risk.select_dtypes(include=[object, np.int64]).columns # create list of categorical names
continuous_names = credit_risk.select_dtypes(exclude=[object, np.int64]).columns # create list of continuous names

# define features (X) and target (y)
credit_risk['TARGET'] = credit_risk['TARGET'].astype(str)
X = credit_risk[colnames] # X value contains all the variables except labels
y = credit_risk['TARGET'] # these are the labe'

# create training test split data sets, with test size of 30% of the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
credit_train = pd.DataFrame(X_train).assign(TARGET = y_train)
credit_test = pd.DataFrame(X_test).assign(TARGET = y_test)

# set up model configuration for Exploratory Data Analysis
s = setup(
    fold=10, # 10 fold cross validation
    data = credit_train, # training data
    silent = True,
    test_data = credit_test, # test data
    target = 'TARGET',
    session_id = 123)

# display basic logistic regression model on raw data
best = compare_models(include = ['lr'])
lr_fit = create_model('lr', fold=5)
plot_model(lr_fit, plot='confusion_matrix',  plot_kwargs = {'percent' : True})
plot_model(lr_fit, 'boundary')

best = compare_models(include = ['lr', 'knn', 'rf', 'dt', 'ada', 'xgboost'],
                      sort = 'F1')

clf1 = setup(data = credit_risk,
             target = 'TARGET',
             fix_imbalance = True,
             normalize = True,
             log_experiment = True,
             remove_multicollinearity = True,
             multicollinearity_threshold = 0.6) #transformation = True

rf = create_model('rf', fold = 2)
plot_model(rf)