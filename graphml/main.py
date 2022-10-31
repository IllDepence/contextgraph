from graphml.ml.classification import get_training_data, split_data
from graphml.ml.models import create_models, cross_validate_models, get_performance_of_cv, evaluate, get_baseline_performance
from graphml.preprocessor.parameters import Parameters

def main(data_path, data_path_description, param, embedding_method,
         with_text_info=False, text_embedding_method="bow",
         realistic_testset=False,
         baseline_only=False, baseline_method="random", baseline_univocal_label="pos"
         ):
    '''
    data_path: path for embedding data of nodes / training data
    data_path_description: path for embedding data of text description of entities
    param: object that stores parameters
    embedding_method: embedding method for nodes, supported are: "node2vec", "gcn", "attri2vec"
    realistic_testset: if to adjust testset as the predefined (realistic) ratio
    with_text_indo: boolean value, controls whether text description will be leveraged
    text_embedding_method: "bow" or "bert", for "bow", dimension of embedding vectors will be reduced using PCA
    baseline_only: boolean value, whether shows performance of baseline or other ML methods
    baseline_method: "random" or "univocal"
    baseline_univocal_label: "pos" or "neg"
    '''

    if baseline_only:
        with_text_info = False
    embeddings = get_training_data(
        data_path=data_path,
        data_path_description=data_path_description,
        param=param,
        embedding_method=embedding_method,
        with_text_info=with_text_info,
        text_embedding_method=text_embedding_method
    )
    X_train, X_test, y_train, y_test = split_data(
        dataframe=embeddings,
        param=param,
        realistic_testset=realistic_testset,
        test_size=0.3
    )
    if baseline_only:
        # acc, pre, re, f1
        performance = get_baseline_performance(y_test,
                                               method=baseline_method,
                                               univocal_label=baseline_univocal_label,
                                               num_average=10,
                                               display=True
                                               )
        return performance
    else:
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
    embedding_method = "rgcn"   # "node2vec"
    text_embedding_method = "bert"
    data_path = "/local/users/ujvxd/env/embeddings_graph_sample_" + embedding_method + ".csv"
    data_path_description = "/local/users/ujvxd/env/embeddings_" + text_embedding_method + ".csv"
    param = Parameters()
    performance = main(
        data_path=data_path,
        data_path_description=data_path_description,
        param=param,
        embedding_method=embedding_method,
        with_text_info=False,
        text_embedding_method=text_embedding_method,
        realistic_testset=False,
        baseline_only=False,
        baseline_method="univocal",
        baseline_univocal_label="pos"
    )