from website_prediction_main import *


#function to run the website prediction, classifier list should be a list
#of strings corresponding to which classifiers to use (options are rfc, mnb, and xgb)
#classifier params is a list of dictionaries each of which can include whichever parameters
#you'd like, but the list must be of equal length to classifier list and params should be in
#the same order as the list i.e. if rfc comes first so should rfc params
def run_website_prediction(data, classifiers, classifier_list, classifier_params, cross_val_list):
    
    if len(classifier_list) != len(classifier_params):
        return "Need to include parameters for each classifier"

    scores = []
    params = []
    class_type = []

    for i in range(len(classifier_list)):
        classifier_list[i] = classifier_list[i].lower()

    data.run_preprocessing()
    x, y = data.get_data()
    
    if 'rfc' in classifier_list:
        rfc_idx = classifier_list.index('rfc')
        rfc_params = classifier_params[rfc_idx]
        rfc_cv = cross_val_list[rfc_idx]
        classifiers.cv_rfc(x, y, rfc_params, rfc_cv, True)
        rfc_score, rfc_params = classifiers.get_best_score_rfc()
        scores.append(rfc_score)
        params.append(rfc_params)
        class_type.append('rfc')

    if 'mnb' in classifier_list:
        mnb_idx = classifier_list.index('mnb')
        mnb_params = classifier_params[mnb_idx]
        mnb_cv = cross_val_list[mnb_idx]
        classifiers.cv_mnb(x, y, mnb_params, mnb_cv, True)
        mnb_score, mnb_params = classifiers.get_best_score_mnb()
        scores.append(mnb_score)
        params.append(mnb_params)
        class_type.append('mnb')

    if 'xgb' in classifier_list:
        xgb_idx = classifier_list.index('rfc')
        xgb_params = classifier_params[xgb_idx]
        xgb_cv = cross_val_list[xgb_idx]
        classifiers.cv_rfc(x, y, xgb_params, xgb_cv, True)
        xgb_score, xgb_params = classifiers.get_best_score_xgb()
        scores.append(xgb_score)
        params.append(xgb_params)
        class_type.append('xgb')


    return scores, params, cass_type



if __name__ == '__main__':
    df_file = r'C:\Users\Tommy\AppData\Local\Programs\Python\Python37\NLP\Website Classification\website_classification.csv'
    df=pd.read_csv(df_file)
    classifiers = Classifiers(['rfc', 'mnb', 'xgb'])
    data = data_processor(df, 'cleaned_website_text', df.Category)

    rfc_params = {
    'max_depth': [20, 60, 100],
    'n_estimators': [200, 400,600]}

    mnb_params = {'alpha':[0.00001,0.0001,0.001,0.01,0.1,1]}

    xgb_params = {'n_estimators':[200,600],'max_depth':[20,50],'learning_rate':[0.1,0.01]}

    scores, params, class_type = run_website_prediction(data, classifiers, ['rfc', 'mnb', 'xgb'], [rfc_params, mnb_params, xgb_params], [5, 5, 3])

    
    
    