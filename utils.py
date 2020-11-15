import pandas as pd
import numpy as np


def print_gridsearch_results(grid_searcher, unique_params):
    # For a given gridsearcher and the relevant params used in grid_search, print the results of the runs
    # Get the data as a pandas DF
    gsearch_results_df = pd.DataFrame(grid_searcher.cv_results_).copy()
    print("Dataframe has shape: %s" % (str(gsearch_results_df.shape)))
    n_trials_grid_search = gsearch_results_df.shape[0]
    print("Number of trials used in grid search: ", n_trials_grid_search)

    # Rearrange row order so it is easy to skim
    gsearch_results_df.sort_values('rank_test_score', inplace=True)
    # Transform param-text to match up with cv_results_ representation
    param_keys = [f"param_{key}" for key in unique_params]
    return(gsearch_results_df[param_keys + ['mean_train_score', 'mean_test_score', 'mean_fit_time', 'rank_test_score']])


def test_on_estimator(pipeline_with_vectorizer, X_test_NF, file_path, transform_fn=lambda x: x):
    X_transformed = transform_fn(X_test_NF)
    yhat_positive_proba = pipeline_with_vectorizer.predict_proba(X_transformed)[:, 1]
    np.savetxt(file_path, yhat_positive_proba)
