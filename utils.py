import pandas as pd
import numpy as np
# Plotting utils
import matplotlib
import matplotlib.pyplot as plt


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


def test_on_estimator(pipeline, X_test_NF, file_path, transform_fn=lambda x: x):
    # For a given estimator pipeline, predict probabilites for a test set and
    # save those results to disk.
    # OPTIONAL: Use a supplied transform_fn to translate data if
    # that transformation isn't included in the original pipeline
    X_transformed = transform_fn(X_test_NF)
    yhat_positive_proba = pipeline.predict_proba(X_transformed)[:, 1]
    np.savetxt(file_path, yhat_positive_proba)


def plot_grid_search(cv_results, grid_param, name_param):
    pass
    # # Get Test Scores Mean and std for each grid search
    # mean_test_score_k = cv_results['mean_test_score']
    # mean_train_score_k = cv_results['mean_train_score']

    # # Plot Grid search scores
    # _, ax = plt.subplots(1, 1)

    # # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    # for idx, val in enumerate(grid_param_2):
    #     ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param + ': ' + str(val))

    # ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    # ax.set_xlabel(name_param_1, fontsize=16)
    # ax.set_ylabel('CV Average Score', fontsize=16)
    # ax.legend(loc="best", fontsize=15)
    # ax.grid('on')


    # Calling Method
plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')
