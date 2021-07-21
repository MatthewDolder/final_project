#!/usr/bin/python

############################################################################
#reference:
#I downloaded the skeleton file from https://github.com/oforero/ud120-projects/tree/python-3.8
#on June 4, 2021
#
#Assignment completed by Matthew Dolder
#on July 7, 2021
############################################################################

#####
#financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

#email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

#POI label: [‘poi’] (boolean, represented as integer)
#######
#%%
import sys
import pickle
sys.path.append("../tools/")
from sklearn.model_selection import train_test_split
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from utilities import replace_nan_with_zero, export_to_csv, simple_scatter, decision_tree_feature_select,run_TEST
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

#################

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#MD: Choose all available features to start
features_list = ['poi','salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', \
    'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', \
    'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', \
    'total_payments','total_stock_value'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

##outliers:  Graph two features as X and Y to demonstrate the TOTAL outlier
simple_scatter(data_dict,['poi','salary','total_stock_value'],'figure1.svg')
del data_dict['TOTAL']
simple_scatter(data_dict,['poi','salary','total_stock_value'],'figure2.svg')
#Much better


#MD: I'm adding two new features based as suggested in the video on lesson 12
#ratio_from = from_messages / from_poi_to_this_person
#ration_to = to_messages / from_this_person_to_poi
for r in data_dict:
    from_messages = data_dict[r]['from_messages'] 
    from_poi = data_dict[r]['from_poi_to_this_person'] 
    ratio_from = 0
    to_messages = data_dict[r]['to_messages'] 
    to_poi = data_dict[r]['from_this_person_to_poi']
    ratio_to = 0
    if (replace_nan_with_zero(from_poi)==0) or (from_messages == 'NaN'):
        ratio_from = 0
    else:
        ratio_from = from_messages / from_poi
    if (replace_nan_with_zero(to_poi) == 0) or (to_messages == 'NaN'):
        ratio_to = 0
    else:
        ratio_to = to_messages / to_poi
    
    data_dict[r]['ratio_from'] = ratio_from
    data_dict[r]['ratio_to'] = ratio_to

#MD: export to csv to investigate the numbers in Excel.  
columns = ['name','salary','to_messages','deferral_payments','total_payments','loan_advances','bonus','email_address', \
        'restricted_stock_deferred','deferred_income','total_stock_value','expenses','from_poi_to_this_person','exercised_stock_options', \
        'from_messages','other','from_this_person_to_poi','poi','long_term_incentive','shared_receipt_with_poi','restricted_stock','director_fees','ratio_from','ratio_to']
export_to_csv('enron_features.csv',data_dict,columns)

#add two new features to feature_list
features_list = ['poi','salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', \
    'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', \
    'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', \
    'total_payments','total_stock_value','ratio_from','ratio_to'] 
data=featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#############################################
#MD:  Try lasso regression from chapter 12
from sklearn.linear_model import Lasso
#Without any parameters, I receve the warning
#ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations
#reference: https://stackoverflow.com/questions/20681864/lasso-on-sklearn-does-not-converge
#reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
regression = Lasso(alpha=1.0,max_iter=100, tol=0.0001)
regression.fit(features,labels)
#Note:  I had to set max_iter extremely high (100000000) to get rid of the convergence warning
#and even then, lasso didn't exclude many features. Probably this isn't a good algorithm. 
print(regression.coef_)
#[ 3.29466735e-07 -0.00000000e+00  1.04138327e-07  7.69123024e-08
#  9.57877227e-08 -7.03062089e-08  7.98076753e-07 -2.15786227e-07
#  2.66892138e-08  9.98700464e-08 -2.32932034e-07 -3.02643671e-07
# -6.12827902e-05  0.00000000e+00 -9.63201044e-06  3.00788306e-04
#  1.52044081e-04 -9.63095297e-08  2.36212071e-07  0.00000000e+00
# -1.09734405e-04]
#I'm setting max_iter back to a low number to improve runtime.  
#############################################

#MD:  Try Decision Tree from chapter 12 mini project
print(decision_tree_feature_select(features,labels,features_list))
#Results show Bonus, Expenses, Other as the best features. 
#{'bonus': 11, 'deferred_income': 2, 'to_messages': 1, 'total_stock_value': 2, 'other': 3, 'expenses': 5, 'exercised_stock_options': 2, 'restricted_stock': 1, 'from_messages': 1, 'shared_receipt_with_poi': 1, 'long_term_incentive': 1, 'ratio_to': 1}
#%%

#MD:  Try some classifiers with default params on these 3 features. 
clf=GaussianNB()
print('GNB:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#GaussianNB is disappointing.  Recall is low. 
#GNB: {'accuracy': 0.806060606060606, 'precision': 0.358974358974359, 'recall': 0.11965811965811966, 'f1': 0.1794871794871795, 'scores': {'tp': 14, 'tn': 518, 'fp': 25, 'fn': 103}}
#%%
clf=DecisionTreeClassifier()
#Try DecisionTree
print('DTC:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#Decision Tree has lower accuracy, but better precision and recall.
#DTC: {'accuracy': 0.7681818181818182, 'precision': 0.3474576271186441, 'recall': 0.3504273504273504, 'f1': 0.3489361702127659, 'scores': {'tp': 41, 'tn': 466, 'fp': 77, 'fn': 76}}
#%%

#Try SVC
clf = SVC(kernel='rbf',C=1000,gamma='auto')
print('SVC_rbf - auto:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#SVC with gamma='auto' is a fail bad
#SVC_rbf - auto: {'accuracy': 0.8227272727272728, 'precision': 0, 'recall': 0, 'f1': 0, 'scores': {'tp': 0, 'tn': 543, 'fp': 0, 'fn': 117}}

clf = SVC(kernel='rbf',C=1000,gamma='scale')
print('SVC_rbf - scale:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#gamma = scale is better
#SVC_rbf - scale: {'accuracy': 0.7803030303030303, 'precision': 0.325, 'recall': 0.2222222222222222, 'f1': 0.2639593908629442, 'scores': {'tp': 26, 'tn': 489, 'fp': 54, 'fn': 91}}

#try linear
#clf = SVC(kernel='linear')
#print('SVC_linear:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#linear kernel is very slow.  I had to cancel.  Commenting this out so it doesn't crash. 

#try linearSVC
#clf = LinearSVC()
#print('LinearSVC:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#I need to scale the values to use Linear.  I'll come back to this later.  Commenting out so it doesn't show warnings 


#MD: Can we find more outliers by graphing the chosen features? 
simple_scatter(data_dict,['poi','bonus','expenses'],'figure3.svg')
simple_scatter(data_dict,['poi','bonus','other'],'figure4.svg')
#I see two outliers on the bonus axis and one on the other axis
#try removing those
data_small = data_dict.copy()
del data_small['LAY KENNETH L']  #bonus
del data_small['LAVORATO JOHN J']  #bonus
del data_small['FREVERT MARK A']   #other
simple_scatter(data_small,['poi','bonus','expenses'],'figure5.svg')
simple_scatter(data_small,['poi','bonus','other'],'figure6.svg')
# I do see some groupings.  Let's try some classifiers on the smaller dataset. 

clf=GaussianNB()
print('GNB:',run_TEST(clf,data_small,['poi','bonus','expenses','other']))
#GNB: {'accuracy': 0.843939393939394, 'precision': 0.33962264150943394, 'recall': 0.20930232558139536, 'f1': 0.2589928057553957, 'scores': {'tp': 18, 'tn': 539, 'fp': 35, 'fn': 68}}
#recall improved from GNB above, but not good enough.  

clf=DecisionTreeClassifier()
print('DTC:',run_TEST(clf,data_small,['poi','bonus','expenses','other']))
#DTC: {'accuracy': 0.8090909090909091, 'precision': 0.3181818181818182, 'recall': 0.4069767441860465, 'f1': 0.35714285714285715, 'scores': {'tp': 35, 'tn': 499, 'fp': 75, 'fn': 51}}
#precision when down after removing "outliers"


#I would like to try graphing the new features created above.  Even though Decision tree classifer didn't find them useful features. 
simple_scatter(data_dict,['poi','bonus','ratio_to'],'figure7.svg')
simple_scatter(data_dict,['poi','bonus','ratio_from'],'figure8.svg')
#There is an outlier in ratio_from.  Looking at the XLS, this is KAMINSKI WINCENTY J
#Try removing it and graph again
data_small = data_dict.copy()
del data_small['KAMINSKI WINCENTY J']  #ratio_from outlier
simple_scatter(data_small,['poi','bonus','ratio_from'],'figure9.svg')
#I don't see much of a grouping.  
clf=GaussianNB()
print('GNB - ratio - small:',run_TEST(clf,data_small,['poi','bonus','ratio_from','ratio_to']))
clf=DecisionTreeClassifier()
print('DTC - ratio - small:',run_TEST(clf,data_small,['poi','bonus','ratio_from','ratio_to']))
#GNB - ratio - small: {'accuracy': 0.845, 'precision': 0.4318181818181818, 'recall': 0.21839080459770116, 'f1': 0.29007633587786263, 'scores': {'tp': 19, 'tn': 488, 'fp': 25, 'fn': 68}}
#DTC - ratio - small: {'accuracy': 0.7466666666666667, 'precision': 0.19626168224299065, 'recall': 0.2413793103448276, 'f1': 0.21649484536082475, 'scores': {'tp': 21, 'tn': 427, 'fp': 86, 'fn': 66}}

#not great.  I'll abandon these "outliers"
data_small.clear()


#returning to LinearSVC()
#If I scale the features, can I get linear results? 
'bonus','expenses','other'
bonus = 0
expenses = 0
other = 0
bon_exp_otr = []
for r in data_dict:
    bonus = replace_nan_with_zero(data_dict[r]['bonus'])
    expenses = replace_nan_with_zero(data_dict[r]['expenses'])
    other = replace_nan_with_zero(data_dict[r]['other'])
    #used by min/max scaler below
    bon_exp_otr.append([bonus,expenses,other])

scale_features = np.array(bon_exp_otr)
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(scale_features)
#add to data
i = 0
for r in data_dict:
    data_dict[r]['scaled_bon'] = rescaled_features[i][0]
    data_dict[r]['scaled_exp'] = rescaled_features[i][1]
    data_dict[r]['scaled_otr'] = rescaled_features[i][2]
    i += 1
features_list.append('scaled_bon')
features_list.append('scaled_exp')
features_list.append('scaled_otr')

clf = LinearSVC()
print('LinearSVC:',run_TEST(clf,data_dict,['poi','scaled_bon','scaled_exp','scaled_otr']))
#LinearSVC: {'accuracy': 0.8212121212121212, 'precision': 0.4444444444444444, 'recall': 0.03418803418803419, 'f1': 0.0634920634920635, 'scores': {'tp': 4, 'tn': 538, 'fp': 5, 'fn': 113}}
#it runs now, but recall is very low.  


#Can I tune for better results? 
data=featureFormat(data_dict, ['poi','bonus','expenses','other'], sort_keys = True)
labels, features = targetFeatureSplit(data)
#reference: Lesson 14 in Udacity
#try var smoothing with GNB
parameters = {'var_smoothing':[1e-1, 1e-20]}
svr = GaussianNB()
clf = GridSearchCV(svr, parameters)
clf.fit(features, labels)
print(clf.best_params_)  #{'var_smoothing': 0.1}
clf = GaussianNB(var_smoothing=0.1)
print('GNB - var_smoothing=1:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#recal is sill very low: 
#GNB - var_smoothing=1: {'accuracy': 0.8121212121212121, 'precision': 0.3870967741935484, 'recall': 0.10256410256410256, 'f1': 0.16216216216216214, 'scores': {'tp': 12, 'tn': 524, 'fp': 19, 'fn': 105}}

#I don't think I can tune much with DTC on this small of a dataset, but I'll try. 
#class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0)[source]
parameters = {'splitter':['best', 'random'],'min_samples_split':[2,5]}
svr = DecisionTreeClassifier()
clf = GridSearchCV(svr, parameters)
clf.fit(features, labels)
print(clf.best_params_)  #{'min_samples_split': 5, 'splitter': 'best'}
clf=DecisionTreeClassifier(min_samples_split=5, splitter='best')
print('DTC:',run_TEST(clf,data_dict,['poi','bonus','expenses','other']))
#About the same as DTC without tuning
#DTC: {'accuracy': 0.7803030303030303, 'precision': 0.3627450980392157, 'recall': 0.3162393162393162, 'f1': 0.3378995433789954, 'scores': {'tp': 37, 'tn': 478, 'fp': 65, 'fn': 80}}

#SVC with scaled params:
data=featureFormat(data_dict, ['poi','scaled_bon','scaled_exp','scaled_otr'], sort_keys = True)
labels, features = targetFeatureSplit(data)
parameters = {'gamma':[1, 1000]}
svr = SVC(kernel='rbf')
clf = GridSearchCV(svr, parameters)
clf.fit(features, labels)
print(clf.best_params_)  #{'gamma': 1}
clf=SVC(kernel='rbf',gamma=1)
print('SVC_rbf - gamma 1:',run_TEST(clf,data_dict, ['poi','scaled_bon','scaled_exp','scaled_otr']))
#total bust: 
#SVC_rbf - gamma 1: {'accuracy': 0.8227272727272728, 'precision': 0, 'recall': 0, 'f1': 0, 'scores': {'tp': 0, 'tn': 543, 'fp': 0, 'fn': 117}}
#%%
############Final Output#################
#MD: I had the best score with DTC and features ['poi','bonus','expenses','other']
my_dataset = data_dict
features_list = ['poi','bonus','expenses','other']

clf=DecisionTreeClassifier()
print('DTC:',run_TEST(clf,my_dataset,features_list))
#DTC: {'accuracy': 0.7681818181818182, 'precision': 0.3474576271186441, 'recall': 0.3504273504273504, 'f1': 0.3489361702127659, 'scores': {'tp': 41, 'tn': 466, 'fp': 77, 'fn': 76}}

test_classifier(clf,my_dataset,features_list)
#DecisionTreeClassifier()
#	Accuracy: 0.76455	Precision: 0.34587	Recall: 0.33100	F1: 0.33827	F2: 0.33387
#	Total predictions: 11000	True positives:  662	False positives: 1252	False negatives: 1338	True negatives: 7748


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# %%
