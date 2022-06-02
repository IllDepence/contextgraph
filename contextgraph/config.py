# papers with code data
pwc_data_dir = '/home/ls3data/datasets/paperswithcode/'
pwc_dsets_fn = 'datasets.json'
pwc_meths_fn = 'methods.json'
pwc_evals_fn = 'evaluation-tables.json'
pwc_dsets_ext_fn = 'datasets_ext.json'
pwc_pprs_fn = 'papers-with-abstracts.json'
pwc_api_base_url = ('https://paperswithcode.com/api/internal/papers/'
                    '?format=json&paperdataset__dataset_id=')

# unarXive data
unarxive_paper_dir = '/opt/unarXive/unarXive-2020/papers/'
unarxive_db_fn = 'refs.db'

# preprocessed graph data
graph_data_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'
# # nodes
graph_dsets_fn = 'datasets.jsonl'
graph_meths_fn = 'methods.jsonl'
graph_modls_fn = 'models.jsonl'
graph_tasks_fn = 'tasks.jsonl'
graph_pprs_fn = 'papers.jsonl'
graph_cntxts_fn = 'contexts_used.jsonl'
# # edges
graph_dsets_to_tasks_fn = 'datasets_to_tasks.csv'
graph_dsets_to_pprs_fn = 'datasets_to_papers.csv'
graph_meths_to_colls_fn = 'methods_to_collections.csv'
graph_meth_areas_fn = 'method_areas.jsonl'
graph_tasks_to_subtasks_fn = 'tasks_to_subtasks.csv'
graph_meths_to_dsets_fn = 'methods_to_datasets.csv'
graph_modls_to_pprs_fn = 'models_to_papers_pre.csv'
graph_meths_to_pprs_fn = 'methods_to_papers.csv'
graph_tasks_to_pprs_fn = 'tasks_to_papers.csv'
graph_modls_to_pprs_pre_fn = 'models_to_papers_pre.csv'
graph_modls_to_pprs_fn = 'models_to_papers.csv'
graph_ppr_to_ppr_fn = 'papers_to_papers.csv'
# # for intermmediate processing steps
graph_tasks_pre_fn = 'tasks_pre.jsonl'
