import time
import joblib
import pandas as pd
import numpy as np
import itertools
from joblib import delayed, Parallel
import multiprocessing
import datetime
from  sklearn.metrics import classification_report
import optuna

def custom_score(y_true, y_pred):
    '''To maximize the precision and recall values of the sell and buy  (-1 , 1)'''
    class_report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0.0)
    try:
        recall_sell, recall_buy = class_report['-1']['recall'], class_report['1']['recall']
        precision_sell,precison_buy = class_report['-1']['precision'], class_report['1']['precision']
        a = precison_buy * recall_buy
        b = precision_sell * recall_sell
    except KeyError as e:
        print(f'KeyError: {e} --CLASS NOT PRESENT')
        return 0.0

    return a * b

start_time = time.perf_counter()

pd.options.display.width = 0
df = pd.read_csv('C:/Users/25473/Documents/Dataframe2/15M_tp_0.001_sl_0.0005.csv')

#shuffle the data
ten_pa = round(len(df) * 0.1)
df = df.sample(frac=1, random_state=42)
validation_set = df.head(ten_pa)
df = df[ten_pa:]
k_best_df = df.copy()

#Check for data quality
k_best_df.replace([np.inf, -np.inf], np.nan, inplace=True)
k_best_df.dropna(axis=0, inplace=True)
print(k_best_df)

if k_best_df.isnull().values.any():
    raise Exception('NaN Values present in dataframe')
else:
    print('Data Pre_processing Complete')

# Corelation Removal
corr_matrix = k_best_df.drop('TRD_Final', axis=1).corr().abs()
corr_df = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corelated_columns = [col for col in corr_df.columns if any(corr_df[col] > 0.95)]
all_columns = corr_df.columns
test  = []

#grouping according to orelations with each other
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
import lightgbm as lgm

x = k_best_df.drop('TRD_Final', axis= 1)
y = k_best_df['TRD_Final']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print('Importance scores are being calculated')
#Importance values
model = lgm.LGBMClassifier(n_jobs=-1)
model.fit(X_train, y_train)
importance = model.feature_importances_
importance_scores = [round(score*1000,3) for score in importance]
values =  dict(zip(x.columns,importance_scores))

#feature deletion (according to importance and corelation values ) __ reduction for KBest
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
x = filtered_df

            # recursive feature eliination
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,StackingClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report,make_scorer

print("RFECV Running")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)# update the train and test set ( features were decreased)
cross_validation = StratifiedKFold(n_splits=10)
rfe = RFECV(cv=cross_validation, n_jobs=-1, estimator=lgm.LGBMClassifier(n_jobs=-1), scoring=make_scorer(custom_score,greater_is_better=True),verbose=1)
rfe.fit(X_train, y_train)

rfe_rankings = list(zip(x.columns,rfe.ranking_))
selected_columns = [feature for feature,rank in rfe_rankings if rank == 1]
print(f'Features selected in rcecv:{selected_features}\nBest number of features : {rfe.n_features_}')
x = pd.DataFrame(k_best_df[selected_columns])# updated features
print(f'Features to use {x.columns}')

#updating training and test sets with current values
filtered_df = pd.DataFrame(k_best_df[selected_columns])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)#Must be updated again

            #hyper-parameter optimization
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, recall_score, precision_score, make_scorer
from  itertools import product
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from xgboost import XGBClassifier

            #Hyperparameter search for XGBoost
import optuna
from optuna import Trial





param_space = {
    'n_estimators': Integer(100, 700),
    'max_depth': Integer(3, 15),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(5, 20),
    'max_features': Categorical(['sqrt', 'log2', 0.5]),
    'bootstrap': Categorical([True, False]),
    'criterion': Categorical(['gini', 'entropy'])
}
bayes_search = BayesSearchCV(estimator=RandomForestClassifier(n_jobs=-1), scoring=make_scorer(custom_score,greater_is_better=True),
                     search_spaces=param_space, cv=StratifiedKFold(10), n_jobs=-1,verbose=1)

bayes_search.fit(X_train,y_train)
print(f'\n best parameters (rfc) = {bayes_search.best_params_}')

#change from gbc to xboost
import optuna
from optuna import Trial

gbc_best = None
best_score = 0.000000001  # Initialize with a very low score

def objective(trail: Trial):

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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = custom_score(y_test, y_pred)

    #Store the best model to avoid retraining model

    global gbc_best
    global  best_score
    if score > best_score:
        best_score = score
        gbc_best = model
    return score

study = optuna.create_study()
study.optimize(objective, n_trials = 20)
print(f"Optuna ( Bayes Search ) best params: {study.best_params}")

#Validation set for RFC
rfc_best = bayes_search.best_estimator_
x_validation, y_validation  = validation_set[x.columns], validation_set['TRD_Final']

print(f'rfc Validation\n{classification_report(y_validation,rfc_best.predict(x_validation))}')

#Validation set for GBC
print(f'gbc Validation\n{classification_report(y_validation,gbc_best.predict(x_validation))}')
joblib.dump(gbc_best,'gbc-sl-tp.pkl')
joblib.dump(rfc_best,'rfc-sl-tp.pkl')

# Probability threshold selection-- choose the correct model ( with the correct amount)
def thresh(probabilities_y,model_type):
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
    #Save  (predictive thresholds) to  to file
    path = f'Z:/files/{model_type}'
    scoree = pd.DataFrame(scores_sept)
    scoree.to_csv(path, index=False)

thresh(rfc_best.predict_proba(X_test),'rfc')
thresh(gbc_best.predict_proba(X_test),'gbc')

end_time = time.perf_counter()
print(f'time taken {end_time-start_time}')

x = input('put in the paramters for model stacking ')

#Carry out the threshold test
from sklearn.ensemble import StackingClassifier
estimators = [
    ('rf', rfc_best()),
    ('gbc', gbc_best())
]
stc = StackingClassifier(
    estimators,final_estimator=RandomForestClassifier(n_jobs=-1), n_jobs=-1,
)
print('Model Stacking.....')
stc.fit(X_train,y_train)
y_pred = stc.predict(X_test)
print(classification_report(y_test,y_pred))
print(f'Validation (stack)\n{classification_report(y_validation,stc.predict(x_validation))}')
joblib.dump(stc, 'stacked_model.pkl')
thresh(y_pred,'stack_model')


# what format will scores be stored as ??


# how are we going to implent the eraly stoping measures (decrtease the training time )--------------------
#should i revert to time series?
#saving models in a particular dir




