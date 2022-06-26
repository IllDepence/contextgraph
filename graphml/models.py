from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import mean


def create_models():
    clfs = {}
    clfs["LogisticRegression"] = LogisticRegression(dual=False,
                                                    C=0.5,
                                                    solver='sag',
                                                    max_iter=1000
                                                    )
    clfs["SVMLinear"] = LinearSVC(random_state=1,
                                  dual=False,
                                  C=0.5,
                                  max_iter=10000
                                  )
    clfs["DecisionTree"] = DecisionTreeClassifier()
    clfs["RandomForest"] = RandomForestClassifier()
    clfs["NaiveBayes"] = GaussianNB()
    return clfs


def cross_validate_models(model_dict, training_data, training_label,
                          cv=None, return_estimator=True
                          ):
    results = {}
    scoring = ['accuracy', 'recall', 'precision', 'f1']
    for model in model_dict:
        print(f'currently fitting the model: {model}')
        results[model] = cross_validate(model_dict[model],
                                        training_data,
                                        training_label,
                                        cv=cv,
                                        scoring=scoring,
                                        return_estimator=return_estimator
                                        )
    return results


def get_performance_of_cv(results, display=True):
    '''display the averaged values (over cross validatioin sets) of all metrics'''

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
    evaluation = {}
    for clf in results_cv:
        accuracies, recalls, precisions, f1 = [], [], [], []
        for estimator in results_cv[clf]["estimator"]:
            pred = estimator.predict(test_data)
            accuracies.append(accuracy_score(test_label, pred))
            recalls.append(recall_score(test_label, pred))
            precisions.append(precision_score(test_label, pred))
            f1.append(f1_score(test_label, pred))

        evaluation[clf] = {}
        evaluation[clf]["accuracy"] = mean(accuracies)
        evaluation[clf]["recall"] = mean(recalls)
        evaluation[clf]["precision"] = mean(precisions)
        evaluation[clf]["f1-score"] = mean(f1)
        if display:
            print(f'Performance of {clf} on test data:')
            print(f"\tAccuracies: {mean(accuracies)}")
            print(f"\tRecalls: {mean(recalls)}")
            print(f"\tPrecisions: {mean(precisions)}")
            print(f"\tF1-Scores: {mean(f1)}")
            print('-' * 35)
    return evaluation
