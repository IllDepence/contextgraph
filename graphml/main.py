from classification import get_training_data, split_data
from models import create_models, cross_validate_models, get_performance_of_cv, evaluate
from parameters import Parameters

def main(data_path_node2vec, data_path_bow, param, process_pattern, with_text_info):

    embeddings = get_training_data(
        data_path_node2vec=data_path_node2vec,
        data_path_bow=data_path_bow,
        param=param,
        process_pattern=process_pattern,
        with_text_info=with_text_info
    )
    X_train, X_test, y_train, y_test = split_data(
        dataframe=embeddings,
        param=param,
        test_size=0.2
    )
    classifiers = create_models()
    results = cross_validate_models(
        model_dict=classifiers,
        training_data=X_train,
        training_label=y_train
    )
    performance_cv = get_performance_of_cv(results)
    evaluation = evaluate(
        results_cv=results,
        test_data=X_test,
        test_label=y_test,
        display=True
    )
    return performance_cv, evaluation


if __name__ == "__main__":
    data_path_node2vec = "/local/users/ujvxd/env/embeddings_graph_sample_new.csv"
    data_path_bow = "/local/users/ujvxd/env/embeddings_bow.csv"
    param = Parameters()
    performance_cv, evaluation = main(
        data_path_node2vec=data_path_node2vec,
        data_path_bow=data_path_bow,
        param=param,
        process_pattern="avg",
        with_text_info=True)