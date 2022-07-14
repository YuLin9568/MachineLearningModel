import pandas
import pickle
import time
import numpy as np

from sklearn import ensemble, preprocessing, metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate

##Training the data by SVM
##If kenerl='rbf' there are two important paramter to control performance
##gamma: Higher gamma means lower redius of gaussian >> overfitting
##c: Higher c means more training points in violation (inside the margin or wrongly classified) >> lower regularization >> overfitting
##argv1: list of data (n_sample, n_feature)
##argv2: list of result (n_sample)
##argv3: list of feature 
##argv4: the file to store the model parameter

def train_model_svm_tune(data_model, result_model, path_dump_model):

    #train_x, test_x, train_y, test_y = train_test_split(model_X, model_Y, test_size = 0.2)
    train_x, test_x = data_model['train'], data_model['test']
    train_y, test_y = result_model['train'], result_model['test']
    gamma = 1 / (len(train_x[0]) * np.var(train_x))
    print(f'Perform SVM by default parameter c=1.0, gamma={gamma}')
    svc_cl = SVC(kernel='rbf')
    svc_fit = svc_cl.fit(train_x, train_y)
    result_test_predicted = svc_cl.predict(test_x) 
    pred = metrics.classification_report(test_y, result_test_predicted)
    accuracy_score = metrics.accuracy_score(test_y, result_test_predicted)
    print(pred)
    
    dict_accuracy = {}
    max_accuracy = accuracy_score
    ret_model = svc_cl
    cs = np.logspace(3, -3, 4)
    gammas = np.logspace(-3, 3, 4) * gamma
    for c in cs:
        for gamma in gammas:
            print(f'Perform SVM by c={c}, gamma={gamma}')
            svc_cl = SVC(kernel='rbf', C=c, gamma=gamma)
            svc_fit = svc_cl.fit(train_x, train_y)
            result_test_predicted = svc_cl.predict(test_x)
            pred = metrics.classification_report(test_y, result_test_predicted)
            accuracy_score = metrics.accuracy_score(test_y, result_test_predicted)
            dict_accuracy[f'{c}-{gamma}'] = accuracy_score
            print(f'{c}-{gamma}: {accuracy_score}')
            if dict_accuracy[f'{c}-{gamma}'] > max_accuracy: 
                print(f'optimized parameter: {c} {gamma}')
                print(pred) 
                max_accuracy = dict_accuracy[f'{c}-{gamma}'] 
                ret_model = svc_cl
        if max_accuracy == 1.0:
            print('Early stopping because of enough accuracy')
            break

    if ret_model: 
        pickle.dump(ret_model, open(path_dump_model, 'wb'))  
        print(f'Dump model into SVM {path_dump_model}')
    
    else:
        print('There is no optimized model')
    
    return ret_model, dict_accuracy    

def train_model_svm_cv(data_model, result_model, path_dump_model):

    gamma = 1 / (len(data_model[0]) * np.var(data_model))
    svc_cl = SVC(kernel='rbf')
    cv_results = cross_validate(svc_cl, data_model, result_model, cv=5, return_estimator=True)
    scores = cv_results['test_score']
    max_accuracy = np.mean(scores)
    print(f'Perform SVM by default parameter c=1.0, gamma={gamma}, accuracy: {max_accuracy}')
    print(scores)
    ret_model = False
    for i, s in enumerate(scores):
        if s == max(scores):
            ret_model = cv_results['estimator'][i]
    ##The parameter from underfitting to overfitting
    cs = np.logspace(3, -3, 4)
    gammas = np.logspace(-3, 3, 4) * gamma
    dict_accuracy = {}
    for gamma in gammas:
        if max_accuracy == 1.0:
            print('Early stopping because of enough accuracy')
            break
        pre_accuracy = 0.0  ##store the accuracy of previous model
        for c in cs:
            svc_cl = SVC(kernel='rbf', C=c, gamma=gamma)
            cv_results = cross_validate(svc_cl, data_model, result_model, cv=5, n_jobs=5, scoring='accuracy', return_estimator=True)
            scores = cv_results['test_score']
            accuracy_score = np.mean(scores)
            dict_accuracy[f'{c}-{gamma}'] = accuracy_score
            print(f'Perform SVM by c={c}, gamma={gamma}, accuracy: {accuracy_score}')
            if accuracy_score < pre_accuracy * 0.85:
                print('Directly arrive next model because of too much regularization')
                break
            pre_accuracy = accuracy_score
            if dict_accuracy[f'{c}-{gamma}'] > max_accuracy: 
                print(f'optimized parameter: {c} {gamma}')
                print(scores)
                max_accuracy = dict_accuracy[f'{c}-{gamma}'] 
                ret_model = svc_cl
                for i, s in enumerate(scores):
                    if s == max(scores):
                        ret_model = cv_results['estimator'][i]

    if ret_model: 
        pickle.dump(ret_model, open(path_dump_model, 'wb'))  
        print(f'Dump model into SVM {path_dump_model}')
        result_test_predicted = ret_model.predict(data_model)
        pred = metrics.classification_report(result_model, result_test_predicted)
        print(pred)
    
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
    path_dump_model = 'svm_test.sav'     ##The text to store the model parameter 
    start_time = time.time()
    train_model_svm_cv(data_model, result_model, feature_list, path_dump_model)
    time_duration = time.time() - start_time
    print("preprocess time: " + str(time_duration) + "s.")
