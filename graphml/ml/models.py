import pandas as pd
from scipy import stats
from scipy.stats import mode
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix
from statistics import mean


def create_models():
    '''
    Simply initialize the 7 predefined models and store
    them into a dict, which will be returned as output
    '''
    clfs = {}
    clfs["LogisticRegression"] = LogisticRegression(
        dual=False,
        C=0.5,
        solver='sag',
        max_iter=1000
    )
    clfs["SVMLinear"] = LinearSVC(
        random_state=1,
        dual=False,
        C=0.5,
        max_iter=10000
    )
    clfs["SVMNonLinear"] = SVC(
        kernel="rbf",
        C=0.5,
        gamma="scale",
        random_state=42
    )
    clfs["DecisionTree"] = DecisionTreeClassifier()
    clfs["RandomForest"] = RandomForestClassifier()
    clfs["NaiveBayes"] = GaussianNB()
    clfs["MLPClassifier"] = MLPClassifier(solver="adam")
    return clfs


def cross_validate_models(
        model_dict,
        training_data,
        training_label,
        cv=None,
        return_estimator=True
):
    '''
    Using 5-fold cross validation by default to fit the
    7 models stored in model_dict. Estimator for every
    cross validation will be returned
    '''
    results = {}
    scoring = ['accuracy', 'recall', 'precision', 'f1']
    for model in model_dict:
        print(f'currently fitting the model: {model}')
        results[model] = cross_validate(
            model_dict[model],
            training_data,
            training_label,
            cv=cv,
            scoring=scoring,
            return_estimator=return_estimator
        )
    return results


def get_performance_of_cv(results, display=True):
    '''
    display the averaged values (over cross validatioin sets)
    of all 4 metrics
     '''

    performance = {}
    for clf in results:
        performance[clf] = {}
        performance[clf]["accuracies"] = mean(results[clf]['test_accuracy'])
        performance[clf]["recalls"] = mean(results[clf]['test_recall'])
        performance[clf]["precisions"] = mean(results[clf]['test_precision'])
        performance[clf]["f1-scores"] = mean(results[clf]['test_f1'])
        if display:
            print(f'Performance of {clf} during cross validation:')
            print(f"\tAccuracies: {mean(results[clf]['test_accuracy'])}")
            print(f"\tRecalls: {mean(results[clf]['test_recall'])}")
            print(f"\tPrecisions: {mean(results[clf]['test_precision'])}")
            print(f"\tF1-Scores: {mean(results[clf]['test_f1'])}")
            print('-' * 35)
    return performance


def evaluate(results_cv, test_data, test_label, display=False):
    '''
    Evaluate the performance on test dataset. The results are
    still averaged across 5 estimators
    '''
    evaluation = {}
    for clf in results_cv:
        accuracies, recalls, precisions, f1 = [], [], [], []
        predictions = []
        for estimator in results_cv[clf]["estimator"]:
            pred = estimator.predict(test_data)
            predictions.append(pred)
            accuracies.append(accuracy_score(test_label, pred))
            recalls.append(recall_score(test_label, pred))
            precisions.append(precision_score(test_label, pred))
            f1.append(f1_score(test_label, pred))
        prediction = mode(predictions)[0][0]
        conf_matrix = pd.DataFrame(
            confusion_matrix(test_label, prediction),
            index=["true_neg", "true_pos"],
            columns=["predicted_neg", "predicted_pos"]
        )
        evaluation[clf] = {}
        evaluation[clf]["accuracy"] = mean(accuracies)
        evaluation[clf]["recall"] = mean(recalls)
        evaluation[clf]["precision"] = mean(precisions)
        evaluation[clf]["f1-score"] = mean(f1)
        evaluation[clf]["confusion_matrix"] = conf_matrix
        '''
        Claimer: the four usual performance measures displayed
        are little bit different from calculated results from
        confusion matrix, that is because the the Confusion Matrix
        is constructed based on a "Voting" methods from all estimators,
        while performance measures displayed here are calculated by the
        average of performance measures of all estimators
        (details see code above)
        '''
        if display:
            print(f'Performance of {clf} on test data:')
            print(f"\tAccuracies: {mean(accuracies)}")
            print(f"\tRecalls: {mean(recalls)}")
            print(f"\tPrecisions: {mean(precisions)}")
            print(f"\tF1-Scores: {mean(f1)}")
            print(f"\tConfusion Matrix:\n{conf_matrix}")
            print('-' * 35)
    return evaluation


def get_baseline_performance(
        dataseries,
        method,
        univocal_label="pos",
        num_average=10,
        display=True
):
    '''
    Evaluate the reuslts of baseline

    Parameters
    ----------
    dataseries: a serie of true labels
    method: str, supported are "random" / "univocal", details are
            given in ../main.py
    univocal_label: str, label to be used if method is univocal
    num_average: time of averaging the results to get rid of noise
            for using random method
    display: default True, performance of baseline method is to
            be displayed
    '''
    if method == "random":
        accuracies, precisions, recalls, f1s = [], [], [], []
        for i in range(num_average):
            random_labels = stats.bernoulli.rvs(0.5, size=dataseries.shape[0])
            accuracies.append(
                accuracy_score(dataseries.values, random_labels)
            )
            precisions.append(
                precision_score(dataseries.values, random_labels)
            )
            recalls.append(
                recall_score(dataseries.values, random_labels)
            )
            f1s.append(
                f1_score(dataseries.values, random_labels)
            )
        acc = mean(accuracies)
        pre = mean(precisions)
        re = mean(recalls)
        f1 = mean(f1s)
    elif method == "univocal":
        if univocal_label == "pos":
            labels = [1]*dataseries.shape[0]
        else:
            labels = [0]*dataseries.shape[0]
        pre = precision_score(dataseries.values, labels)
        acc = accuracy_score(dataseries.values, labels)
        re = recall_score(dataseries.values, labels)
        f1 = f1_score(dataseries.values, labels)
    else:
        raise Exception(f'method {method} for performing Baseline '
                        f'prediction not supported, '
                        f'please use one from "random" or "univocal"')
    if display:
        print(f'Performance of Baseline prediction '
              f'using {method} method on test data:')
        print(f"\tAccuracy: {acc}")
        print(f"\tRecall: {re}")
        print(f"\tPrecision: {pre}")
        print(f"\tF1-Score: {f1}")
    return acc, pre, re, f1
