import pickle
import time
import xgboost as xgb
import numpy as np

from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_validate

##Training the data by random forest
##argv1: list of data (n_sample, n_feature)
##argv2: list of result (n_sample)
##argv3: list of feature 
##argv4: the file to store the model parameter
def train_model_xgb(data_model, result_model, feature_list, path_dump_model):
    model_X = data_model
    model_Y = result_model

    train_x, test_x, train_y, test_y = train_test_split(model_X, model_Y, test_size = 0.2)
    xgb_cl = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss')
    #forest = ensemble.RandomForestClassifier(n_estimators = 300)
    xgb_fit = xgb_cl.fit(train_x, train_y)

    pickle.dump(xgb_cl, open(path_dump_model, 'wb'))
    accuracy, pred, feature_importances = verify_model(test_x, test_y, feature_list, path_dump_model)

    return accuracy, pred, feature_importances

def train_model_xgb_tune(data_model, result_model, feature_list, path_dump_model):

    #train_x, test_x, train_y, test_y = train_test_split(model_X, model_Y, test_size = 0.2)
    train_x, test_x = data_model['train'], data_model['test']
    train_y, test_y = result_model['train'], result_model['test']
    #performance = []
    dict_accuracy = {}
    max_accuracy = 0.0
    ret_model = False
    for i in [50, 100, 200, 300]:
        #dict_accuracy[i] = []
        for j in [3, 6, 9]:
            xgb_cl = xgb.XGBClassifier(n_estimators=i, max_depth=j, eval_metric='logloss')
            xgb_fit = xgb_cl.fit(train_x, train_y)
            result_test_predicted = xgb_cl.predict(test_x)
            
            pred = metrics.classification_report(test_y, result_test_predicted)
            accuracy_score = metrics.accuracy_score(test_y, result_test_predicted)
            dict_accuracy[f'{i}-{j}'] = accuracy_score
            print(f'{i}-{j}: {accuracy_score}')
            if dict_accuracy[f'{i}-{j}'] > max_accuracy: 
                print(f'optimized parameter: {i} {j}')
                print(pred) 
                max_accuracy = dict_accuracy[f'{i}-{j}'] 
                ret_model = xgb_cl
        if max_accuracy == 1.0:
            print('Early stopping because of enough accuracy')
            break

    if ret_model: 
        pickle.dump(ret_model, open(path_dump_model, 'wb'))  
        print(f'Dump model into {path_dump_model} and show importances')
        importances = list(ret_model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        for pair in feature_importances[:10]:
            if pair[-1] == 0.0:
                break
            print('Variable: {:20} Importance: {:.4f}'.format(*pair))
    
    else:
        print('There is no optimized model')
    
    return ret_model, dict_accuracy    


def train_model_xgb_cv(data_model, result_model, feature_list, path_dump_model):
    print('Search the optimized parameter by cross validation')
    dict_accuracy = {}
    max_accuracy = 0.0
    ret_model = False
    for i in [50, 100, 300, 500]:
        #dict_accuracy[i] = []
        for j in [3, 6, 9]:
            xgb_cl = xgb.XGBClassifier(n_estimators=i, max_depth=j, eval_metric='logloss')
            cv_results = cross_validate(xgb_cl, data_model, result_model, cv=5, scoring='accuracy', return_estimator=True)
            scores = cv_results['test_score']
            dict_accuracy[f'{i}-{j}'] = np.mean(scores)
            print(f'{i}-{j}: {np.mean(scores)}')
            if dict_accuracy[f'{i}-{j}'] > max_accuracy:
                print(f'optimized parameter: {i}')
                print(scores)
                max_accuracy = dict_accuracy[f'{i}-{j}']
                for k, s in enumerate(scores):
                    if s == max(scores):
                        ret_model = cv_results['estimator'][k]
        if max_accuracy >= 1.0:
            print('Early stopping because of enough accuracy')
            break

    if ret_model: 
        pickle.dump(ret_model, open(path_dump_model, 'wb'))  
        print(f'Dump model into {path_dump_model} and show importances')
        result_test_predicted = ret_model.predict(data_model)
        pred = metrics.classification_report(result_model, result_test_predicted)
        print(pred)
        importances = list(ret_model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        for pair in feature_importances[:10]:
            if pair[-1] == 0.0:
                break
            print('Variable: {:20} Importance: {:.4f}'.format(*pair))
    
    else:
        print('There is no optimized model')
    
    return ret_model, dict_accuracy    

def verify_model(data_test, result_test, feature_list, path_load_model):
    forest = pickle.load(open(path_load_model, 'rb'))    
    result_test_predicted = forest.predict(data_test)

    accuracy = metrics.accuracy_score(result_test, result_test_predicted)
    pred  = metrics.classification_report(result_test, result_test_predicted)
    #pred  = metrics.classification_report(test_y_predicted, test_y)
    #print(accuracy)
    print(pred)
    
    importances = list(forest.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    for pair in feature_importances[:10]:
        if pair[-1] == 0.0:
            break
        print('Variable: {:20} Importance: {:.4f}'.format(*pair))

    return float(accuracy), pred, feature_importances

##Enter list of data and list of result to output the performance
def test_data(list_data, list_result, path_load_model):

    forest = pickle.load(open(path_load_model, 'rb'))
    list_result_predicted = forest.predict(list_data)

    accuracy = metrics.accuracy_score(list_result, list_result_predicted)
    pred  = metrics.classification_report(list_result_predicted, list_result)
    print(accuracy)
    print(pred)
    
    return list_result_predicted, accuracy, pred 

if __name__ == "__main__":
    data_model = []
    result_model = []
    feature_list = []
    path_dump_model = 'xgb_test.sav'     ##The text to store the model parameter 
    start_time = time.time()
    train_model_xgb_cv(data_model, result_model, feature_list, path_dump_model)
    time_duration = time.time() - start_time
    print("preprocess time: " + str(time_duration) + "s.")
