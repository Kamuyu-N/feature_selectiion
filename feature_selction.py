import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import delayed, Parallel
import multiprocessing
import datetime

#original_data
original = pd.read_csv("C:/Users/25473/Downloads/EURUSD_4H.csv", sep= '\t') # original Storage location

pd.options.display.width = 0
df = pd.read_csv('C:/Users/25473/Documents/DataFrame/tp_0.003_sl_0.0015.csv')
df['DATETIME'] = pd.to_datetime(original['<TIME>'], format='%H:%M:%S').dt.time
time_from = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
time_to = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()
df['DATETIME'] = [np.nan if x > time_from and x < time_to else x for x in df['DATETIME'] ]

# df['TRD_Final'] = [0 if x == -1 else x for x in df['TRD_Final']]
df.dropna(axis=0, inplace=True)
df.drop('DATETIME', axis=1, inplace=True) # has no need ( was only used for filtering the wrong dates)

validation_set = df.tail(2190)
df = df[:len(validation_set)]

#Check for data quality
k_best_df = df.copy()
k_best_df.replace([np.inf, -np.inf], np.nan, inplace=True)
k_best_df.dropna(axis=0, inplace=True)

if k_best_df.isnull().values.any(): # Check if any NaN values exist in the entire DataFrame
    raise Exception('NaN Values present in dataframe')
else:
    print('Data Pre_processing Complete')

# Corelation Removal
corr_matrix = df.corr().abs()
corr_df = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corelated_columns = [col for col in corr_df.columns if any(corr_df[col] > 0.85)]
all_columns = corr_df.columns

test  = []
#(X,Y)
for index,col in enumerate(corelated_columns):
    temp = (corr_df[col]).tolist()
    index_loc = [temp.index(val) for val in temp if val > 0.95]
    temp_columns = [all_columns[i] for i in index_loc]
    temp_columns.append(col)
    test.append(temp_columns)

# use test when you already have them
columns_to_find = list(set([y for x in test for y in x]))

#filter method- using the importance scores
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score,classification_report

gbc = GradientBoostingClassifier()
x = k_best_df.drop('TRD_Final', axis= 1)
y = k_best_df['TRD_Final']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#Importance values
model = RandomForestClassifier(verbose=1, n_jobs=-1)
model.fit(X_train, y_train)

importance = model.feature_importances_
importance_scores = [round(score*1000,3) for score in importance]
values =  dict(zip(x.columns,importance_scores))
cpu_no = multiprocessing.cpu_count()

#feature deletion ( accorsing to their importance scores )
def reduction(corr_list):
    best_features = []
    for vals in corr_list:
        temp_a = []
        for sub in vals:
            temp_a.append(f'{values[f"{sub}"]}')
        index = temp_a.index(max(temp_a))
        best_features.append(vals[index])

    return list(set(best_features))

best_columns = reduction(corr_list=test)# Best columns from corelation
x.drop(corelated_columns, axis=1, inplace=True)

filtered_x = list(set([*x,*best_columns]))
filtered_df = pd.DataFrame(k_best_df[filtered_x])
num_of_classes = k_best_df['TRD_Final'].value_counts()


prec_list = []
for index,k in enumerate(range(5, 200,2), start=5):
    selector = SelectKBest(mutual_info_classif,k=k)
    selector.fit(X_train,y_train)# get the importance values

    x_train_v2 = selector.transform(X_train) # only contains the k best features
    x_test_v2 = selector.transform(X_test)

    gbc.fit(x_train_v2,y_train)
    pred = gbc.predict(x_test_v2)
    precis = precision_score(y_test,pred, average='weighted')
    print(f'P score is {precis}')

    prec_list.append([index, ])

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,StackingClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report


             # recursive feature elimination
cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
rfe = RFECV(cv=cross_validation, n_jobs=-1, estimator=RandomForestClassifier(n_jobs=-1), scoring='precision_weighted',verbose=1)

rfe.fit(filtered_df, y)
print(f'Best number of features : {rfe.n_features_}')



rankings = rfe.ranking_

rfe_rankings = list(zip(filtered_x,rankings))
selected_columns = [feature for feature,rank in rfe_rankings if rank == 1]

#Test
filtered_df = pd.DataFrame(k_best_df[selected_columns])
x_validation, y_validation = filtered_df.tail(1000), y.tail(1000) #Validation Set

print(f'Values in the dataframe: {y.value_counts()}')
filtered_df = filtered_df.iloc[:-1000]
y = y.iloc[:-1000]

#Model training
X_train, X_test, y_train, y_test = train_test_split(filtered_df, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(verbose=1, n_jobs=-1,  n_estimators=500, max_depth=8, max_features=0.5, min_samples_split=10, min_samples_leaf=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Test data\n {classification_report(y_test,y_pred)}")

#Validation Data
print({classification_report(y_true=y_validation,y_pred=model.predict(x_validation), zero_division=0.0)})


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, recall_score, precision_score, make_scorer
from  itertools import product
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

def custom_score(y_true, y_pred):
    '''To maximize the precision and recall values of the sell and buy  (-1 , 1)'''
    class_report = classification_report(y_true, y_pred, output_dict=True)
    recall_sell, recall_buy = class_report['-1.0']['recall'], class_report['1.0']['recall']
    precision_sell,precison_buy = class_report['-1.0']['precision'], class_report['1.0']['precision']
    a = precison_buy * recall_buy
    b = precision_sell * recall_sell

    return a * b

param_space = {
    'n_estimators': Integer(50, 700),
    'max_depth': Integer(3, 15),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(5, 20),
    'max_features': Categorical(['sqrt', 'log2', 0.5]),
    'bootstrap': Categorical([True, False]),
    'criterion': Categorical(['gini', 'entropy'])
}

bayes_search = BayesSearchCV(estimator=RandomForestClassifier(n_jobs=-1), scoring=make_scorer(custom_score, greater_is_better=True),
                     search_spaces=param_space, cv=StratifiedKFold(10, shuffle=True, random_state=42), n_jobs=-1,verbose=1)

bayes_search.fit(X_train,y_train)
print(f'\n best parameters (rfc) = {bayes_search.get_params()}')


#Validation Set( use best prev params)
pred_validation = bayes_search.best_estimator_.p(x_validation)
print('Validation (Rfc)\n\n')
print(classification_report(y_validation, pred_validation))

import optuna
from optuna import Trial
def optimize(trail:Trail):
    n_estimators = trail.suggest_int('n_estimators',100,700)

    if n_estimators < 100:
        learning_rate = trail.suggest_float('learning_rate', 0.15,0.4)
    elif n_estimators < 400:
        learning_rate = trail.suggest_float('learning_rate', 0.05, 0.1)
    else:
        learning_rate = trail.suggest_float('learning_rate', 0.01, 0.05)

    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=trail.suggest_int('max_depth',3,10),
        min_samples_leaf= trail.suggest_int('min_samples_leaf', 2,10),
        min_samples_split=trail.suggest_int('min_samples_split', 2,10),
        max_features=trail.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        subsample=trail.suggest_float('subsample', 0.5, 1.0),
    )

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    #Scoring metrics
    class_report = classification_report(y_true, y_pred)
    recall_sell, recall_buy = class_report['-1.0']['recall'], class_report['1.0']['recall']
    precision_sell, precison_buy = class_report['-1.0']['precision'], class_report['1.0']['precision']

    return (precison_buy * recall_buy) * (precision_sell * recall_sell)

study = optuna.create_study()
study.optimize(objective, n_trials = 1000)
print(f"Optuna ( Bayes Search ): {study.best_params}")

#Creating all possible combinations to be used
probabilities_y = model.predict_proba(X_test)
thresh_combinations = list(itertools.product(np.arange(0.3,0.78 + 0.02, 0.02), repeat=2))

score_values = {
    'Sell_thresh':[],
    'Buy_thresh':[],
    'Recall_score': [],
    'Precision_score':[],
}

for sell_thresh, buy_thresh in thresh_combinations:
    proy =[]
    for i in probabilities_y:
        if i[0] > sell_thresh:
            proy.append(-1)
        elif i[2] > buy_thresh:
            proy.append(1)
        else:
            proy.append(0)
    my_pred = np.array(proy).astype(float)
    class_report = classification_report(y_test,my_pred, output_dict=True, zero_division=0.0)

    precision_values = [class_report['-1.0']['precision'],class_report['1.0']['precision']]
    recall_values = [class_report['-1.0']['recall'],class_report['1.0']['recall']]

    # Get best threshold values and carry out model stacking
    score_values['Sell_thresh'].append(sell_thresh)
    score_values['Buy_thresh'].append(buy_thresh)
    score_values['Recall_score'].append([class_report['-1.0']['recall'], class_report['1.0']['recall']])
    score_values['Precision_score'].append([class_report['-1.0']['precision'],class_report['1.0']['precision']])

# removal of irrelevant values
recall_to_drop = [True if buy > 0.01 and sell > 0.01 else np.nan  for sell, buy in score_values['Recall_score']]
precision_to_drop = [True if buy > 0.5 and sell > 0.5 else np.nan  for sell, buy in score_values['Precision_score']]
score_values['Recall_drop'], score_values['Precision_drop'] = recall_to_drop, precision_to_drop

score_df = pd.DataFrame(score_values)
score_df.dropna(inplace=True, axis=0)
score_df.drop(['Recall_drop','Precision_drop'],axis=1, inplace=True)

# splitting
recall_sell, recall_buy = [x for x,y in score_df.Recall_score],  [y for x,y in score_df.Recall_score]
precision_sell, precison_buy = [x for x,y in score_df.Precision_score], [y for x,y in score_df.Precision_score]

scores_sept = {
    'Recall_sell': recall_sell,
    'Recall_buy': recall_buy,
    'Precision_sell': precision_sell,
    'Precision_buy': precison_buy
}

quit()

#Model stacking( compare results ) 
from sklearn.ensemble import StackingClassifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=500, random_state=42)),
    ('gbc', GradientBoostingClassifier())
]

stc = StackingClassifier(
    estimators,final_estimator=GradientBoostingClassifier() , n_jobs=-1,
)
print('Model stacking is running')
stc.fit(X_train,y_train)
y_pred = stc.predict(X_test)
print(classification_report(y_test,y_pred))

score_df = pd.DataFrame({**dict(score_df),**scores_sept}).drop(['Recall_score', 'Precision_score'], axis=1).reset_index(drop=True)
print(f'{score_df} \n \n GBC training')

# Search for best Hyper-Parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(5, shuffle=True), scoring='weighted_precision')
grid_search.fit(x, y)

 validation sets
lr = GradientBoostingClassifier()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
quit()

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
param_grid = {
    'max_depth':Integer(5,30),
    'min_samples_split': Integer(10,40),
    'min_samples_leaf': Integer(5,20),
    'max_features': Categorical('sqrt', 'log2', 0.5)
}
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer

# Create a base model
rf = RandomForestClassifier()
stratified_fold = StratifiedKFold(10,shuffle=True, random_state=42)





