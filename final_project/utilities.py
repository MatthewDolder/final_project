#!/usr/bin/python

########################################
#Created by Matthew Dolder on July 13, 2021
#Contains functions to complete tedious tasks
########################################

import sys
sys.path.append("../tools/")
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from feature_format import featureFormat, targetFeatureSplit


def replace_nan_with_zero(value):
    if value == 'NaN':
        return_value = 0
    else:
        return_value = value

    return return_value

def export_to_csv(filename,data_dict,columns):
    
    #MD: export to csv to investigate the numbers in Excel.  
    #reference: https://www.geeksforgeeks.org/python-save-list-to-csv/
    #WINOKUR JR. HERBERT S
    #{'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 84992, 'loan_advances': 'NaN', 'bonus': 'NaN', 
    # 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -25000, 'total_stock_value': 'NaN', 'expenses': 1413, 
    # 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 
    # 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 108579}
    with open (filename,'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(columns)
        for d in data_dict:
            onerow = []
            onerow.append(d)
            for f in columns[1:]:
                myval = data_dict[d][f]
                if myval == 'NaN':
                    myval = 0
                onerow.append(myval)
            #print(onerow)
            write.writerow(onerow)
    file.close


def simple_scatter(data_dict,feature_list,file_name):
    #MD:  accepts data_dictionary, feature_list [poi,x,y], file_name and creates 
    #and creates a scatter plot showing poi and non-poi
    graph_data = featureFormat(data_dict, feature_list, sort_keys = True)
    tmp_is_poi = []
    tmp_not_poi = []
    for d in graph_data:
        if d[0] == 1.0:
            tmp_is_poi.append([d[1],d[2]])
        else:
            tmp_not_poi.append([d[1],d[2]])
    is_poi = np.array(tmp_is_poi)
    not_poi = np.array(tmp_not_poi)
    
    plt.clf()
    plt.scatter(is_poi[:,0], is_poi[:,1], color = "r", label="poi")
    plt.scatter(not_poi[:,0], not_poi[:,1], color = "b", label="not poi")
    plt.legend()
    plt.xlabel(feature_list[1])
    plt.ylabel(feature_list[2])
    plt.savefig(file_name, format='svg')
    #plt.show()  #Shows one outlier in the not poi category
    plt.clf()

def decision_tree_feature_select(features,labels,features_list, test_iter=20):
    ##accepts features, labels
    ##returns a list of feature_importance, feature name, feature index
    #reference: Chapter 12 mini project

    best_features = {}
    while test_iter > 0:    
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=test_iter)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(features_train,labels_train)
        ypred = clf.predict(features_test)

        #acc = metrics.accuracy_score(labels_test,ypred)
        #print(acc)
        important_features = clf.feature_importances_
        feature_names = features_list[1:]  #remove PI
        i = 0
        #return_features = []
        for f in important_features:
            if f > 0.2:
                #print(important_features[i],feature_names[i], i)
                #return_features.append([important_features[i],feature_names[i], i])
                if feature_names[i] in best_features:
                    best_features[feature_names[i]] = best_features[feature_names[i]] + 1
                else:
                    best_features[feature_names[i]] = 1

            i+= 1
        test_iter -= 1
    
    return best_features

def run_TEST(clf,data_dict,feature_list, test_iter = 20):
    #######################
    #accepts data_dictionary and feature list
    #returns accuracy, precision, & recall
    ##MD:  I had trouble getting precision and recall from sklearn on a single run which came close to test_classifer
    #Instead I'm creating multiple runs and calculating precision and recall myself.  
    #I'm using accuracy from the last run rather than computing an average accuracy over the runs. 
    #seems to be good enough. 
    ######################
    data=featureFormat(data_dict, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scores = {'tp':0,'tn':0,'fp':0,'fn':0}

    while test_iter > 0:
        
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=test_iter)
        clf.fit(features_train,labels_train)
        ypred = clf.predict(features_test)

        i = 0
        for y in ypred:
            if y == labels_test[i]: #got a match
                if y == 0:  
                    scores['tn'] = scores['tn'] + 1
                else:  
                    scores['tp'] = scores['tp'] + 1
            else:  #not a match
                if y == 0:
                    scores['fn'] = scores['fn'] + 1
                else:
                    scores['fp'] = scores['fp'] + 1
            i += 1

        test_iter -= 1
    if scores['tp'] == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = scores['tp'] / (scores['tp'] + scores['fp'])
        recall = scores['tp'] / (scores['tp'] + scores['fn'])
        
        #reference: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
        _precision = (1.0/precision) + (1.0/recall)
        f1 = 2.0*(1.0/_precision)
        
    accuracy = (scores['tp'] + scores['tn']) / (scores['tp'] + scores['tn'] + scores['fp'] + scores['fn'])
      
    
    #print(scores)
    #'accuracy': metrics.accuracy_score(labels_test,ypred)

    return({'accuracy': accuracy, \
            'precision': precision, \
            'recall': recall, \
            'f1':f1, \
            'scores':scores})

def count_zeros(data_dict,feature_name):
    #MD:  interates the data_dictionary and returns the number of zeros 
    #for a single feature. 

    cnt = 0

    for d in data_dict:
        if replace_nan_with_zero(data_dict[d][feature_name]) == 0:
            cnt +=1 
    return cnt

        
