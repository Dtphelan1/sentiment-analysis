import pandas as pd
import numpy as np
import sklearn
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


# Plot one: A look at a single CV
def plot_cv_train_test(cv_results, param_name, param_label, log10=False, log2=False):
    param_values = cv_results[f'param_{param_name}']
    mean_test_score = cv_results['mean_test_score']
    mean_train_score = cv_results['mean_train_score']

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    x_values = param_values.data.astype(np.float64)
    if (log10):
        x_values = np.log10(x_values)
    elif (log2):
        x_values = np.log2(x_values)
    ax.plot(x_values, mean_test_score, 's', label='test set')
    ax.plot(x_values, mean_train_score, 's', label='train set')

    ax.set_title(f"{param_label} Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(param_label, fontsize=16)
    ax.set_ylabel('Balanced Accuracy', fontsize=16)
    ax.legend(loc="best", fontsize=15, box_to_anchor=(1, 1)))

# Two plots: One comparing train and test performance; one comparing all of the fold scores against one another


def plot_cv_folds(cv_results, param_name, param_label, folds, log10 = False, log2 = False):
    _, ax=plt.subplots(1, 1)

    # Get averages
    mean_test_score=cv_results['mean_test_score']
    mean_train_score=cv_results['mean_train_score']

    # Get x_values
    param_values=cv_results[f'param_{param_name}']
    x_values=param_values.data.astype(np.float64)
    if (log10):
        x_values=np.log10(x_values)
    elif (log2):
        x_values=np.log2(x_values)

    ax.plot(x_values, mean_test_score, '-bs', label = f'average test')
    ax.plot(x_values, mean_train_score, '-rs', label = f'average train')

    # Get overall mean_train_score
    for fold in range(folds):
        fold_scores=cv_results[f'split{fold}_test_score']
        ax.plot(x_values, fold_scores, 's', label = f'{fold}-validation set')

    ax.set_title(f"{param_label} Grid Search Scores", fontsize = 20, fontweight = 'bold')
    ax.set_xlabel(param_label, fontsize = 16)
    ax.set_ylabel('Balanced Accuracy', fontsize = 16)
    ax.legend(loc = "best", fontsize = 15, bbox_to_anchor = (1, 1)))


def _confusion_matrix_idx(estimator, y_train, x_train):
    # Get all of the indices for each quadrant in the confusion matrix
    predictions=estimator.predict(x_train)

    # Confusion matrix
    confusion_matrix=sklearn.metrics.confusion_matrix(y_train, predictions)
    print(confusion_matrix)

    # True positives and true negatives
    true_predictions=np.where(np.equal(y_train, predictions))[0]
    assert(true_predictions.size == confusion_matrix[0][0] + confusion_matrix[1][1])
    true_negatives=np.where(predictions[true_predictions] == 0)[0]
    assert(true_negatives.size == confusion_matrix[0][0])
    true_positives=np.where(predictions[true_predictions] == 1)[0]
    assert(true_positives.size == confusion_matrix[1][1])

    # False positives and false negatives
    false_predictions=np.where(np.not_equal(y_train, predictions))[0]
    assert(false_predictions.size == confusion_matrix[0][1] + confusion_matrix[1][0])
    false_negatives=np.where(predictions[false_predictions] == 0)[0]
    assert(false_negatives.size == confusion_matrix[1][0])
    false_positives=np.where(predictions[false_predictions] == 1)[0]
    assert(false_positives.size == confusion_matrix[0][1])
    return (true_predictions[true_negatives], false_predictions[false_negatives],
            false_predictions[false_positives], true_predictions[true_positives])


def _characterize_examples(x_train_df, x_idx, preprocessor, tokenizer):
    # Tokenize the text
    x_train_text=x_train_df['text'].values[x_idx]
    x_train_website=x_train_df['website_name'].values[x_idx]
    x_train_text_processed=np.array([preprocessor(sent) for sent in x_train_text])
    x_train_text_tokenized=np.array([tokenizer(sent) for sent in x_train_text_processed], dtype='object')

    # Average length of the examples provided
    len_checker=np.vectorize(len)
    avg_length=np.mean(len_checker(x_train_text_tokenized))

    # Website Breakdown of the examples provided
    website_breakdowns={
        "imdb": np.count_nonzero(x_train_website == 'imdb') / len(x_idx),
        "amazon": np.count_nonzero(x_train_website == 'amazon') / len(x_idx),
        "yelp": np.count_nonzero(x_train_website == 'yelp') / len(x_idx)
    }

    # does it do better on sentences without negation words ("not", "didn't", "shouldn't", etc.)?
    # This is what these negations words look like after sklearns default tokenization
    negation_words=[
        'no',
        'not',
        'none',
        'nobody',
        'nothing',
        'neither',
        'nowhere',
        'never',
        'doesn',
        'isn',
        'wasn',
        'shouldn',
        'wouldn',
        'couldn',
        'won',
        #         'can', # ignoring can because it's doubly encoded
        'don',
    ]
    sents_with_negations=0
    for sent in x_train_text_tokenized:
        negations_in_sent=len(list(filter(lambda token: token in negation_words, sent)))
        if negations_in_sent > 0:
            sents_with_negations += 1

    percentage_with_negations=sents_with_negations / len(x_idx)
    # Pretty print the results
    print(f"Average Length of Sentences:    {avg_length}")
    print(f"Breakdown of website names:     {website_breakdowns}")
    print(f"Percentage with negation words: {percentage_with_negations}")
    print('...Examples')

    return (avg_length, website_breakdowns, percentage_with_negations)


def analysis_of_mistakes(pipeline_with_vectorizer, x_train_df, y_train):
    x_train_text=x_train_df['text'].values
    (tn_idx, fn_idx, fp_idx, tp_idx)=_confusion_matrix_idx(pipeline_with_vectorizer, y_train, x_train_text)

    tf_preprocessor=pipeline_with_vectorizer[0].build_preprocessor()
    tf_tokenizer=pipeline_with_vectorizer[0].build_tokenizer()

    print("----- False Positives")
    (fp_avg_length, fp_website_breakdowns, fp_percentage_with_negations)=_characterize_examples(x_train_df, fp_idx, tf_preprocessor, tf_tokenizer)
    fp_text=x_train_text[fp_idx]
    print('...Examples')
    print('\n'.join(fp_text[:10]))
    print()
    print()

    print("----- False Negatives")
    (fn_avg_length, fn_website_breakdowns, fn_percentage_with_negations)=_characterize_examples(x_train_df, fn_idx, tf_preprocessor, tf_tokenizer)
    fn_text=x_train_text[fn_idx]
    print('...Examples')
    print('\n'.join(fn_text[:10]))
    print()
    print()

    print("----- True Positives")
    (tp_avg_length, tp_website_breakdowns, tp_percentage_with_negations)=_characterize_examples(x_train_df, tp_idx, tf_preprocessor, tf_tokenizer)
    tp_text=x_train_text[tp_idx]
    print('...Examples')
    print('\n'.join(tp_text[:10]))
    print()
    print()

    print("----- True Negatives")
    (tn_avg_length, tn_website_breakdowns, tn_percentage_with_negations)=characterize_examples(x_train_df, tn_idx, tf_preprocessor, tf_tokenizer)
    tn_text=x_train_text[tn_idx]
    print('...Examples')
    print('\n'.join(tn_text[:10]))
    print()
    print()


# Calling Method
# plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

# Plot two - scores over each fold
# scores_per_fold_K = []

# _, ax = plt.subplots(1, 1)

# # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
# for fold, scores_ in enumerate(scores_per_fold_K):
#     ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

# ax.set_title("Balanced Accuracy", fontsize=20, fontweight='bold')
# ax.set_xlabel(name_param_1, fontsize=16)
# ax.set_ylabel('CV Average Score', fontsize=16)
# ax.legend(loc="best", fontsize=15)
# ax.grid('on')


# ## Failed attempt at blocking each line in the visualization by a second parameter

# cv_results = tf_rf_grid_searcher.cv_results_
# # Two plots: One comparing train and test performance; one comparing all of the fold scores against one another
# # Plot one
# param = list(rf_parameters.keys())[0]
# param_label = "Number of trees in forest frequency"
# param_values = cv_results[f'param_{param}']
# param_range = rf_parameters[param]


# second_param = list(rf_parameters.keys())[1]
# second_p_label = "Max depth of trees"
# second_p_values = cv_results[f'param_{second_param}']
# second_p_range = rf_parameters[second_param]
# print(second_p_values)
# # print(type(second_p_values))
# print(second_p_range)
# # print(second_p_values == second_p_range[0])


# mean_test_score = cv_results['mean_test_score'].reshape(len(param_range), len(second_p_range))
# mean_train_score = cv_results['mean_train_score'].reshape(len(param_range), len(second_p_range))
# print(mean_test_score)
# print(mean_train_score)
# _, ax = plt.subplots(1, 1)

# for i, val in enumerate(second_p_range):
#     ax.plot(param_range, mean_test_score[:, i], '-o', label=f'test set, {val}')
# #     ax.plot(param_range, mean_train_score[:, i], '-o', label=f'train set, {val}')

# ax.set_title(f"{param_label} Grid Search Scores", fontsize=20, fontweight='bold')
# ax.set_xlabel(second_p_label, fontsize=16)
# ax.set_ylabel('Balanced Accuracy', fontsize=16)
# ax.legend(loc="best", fontsize=15)
# ax.grid('on')
