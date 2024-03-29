from random import shuffle

import numpy as np
from sklearn import metrics as sk_metrics


def estimate_optimal_threshold(test_score, y_test, pos_label=1, nq=100, val_ratio=.2):
    # Generate indices the testscore
    n = len(test_score)
    idx = list(range(n))
    shuffle(idx)
    idx = np.array(idx)

    # split score in test and validation
    n_test = int(n * (1 - val_ratio))

    score_t = test_score[idx[:n_test]]
    y_t = y_test[idx[:n_test]]
    score_v = test_score[idx[n_test:]]
    y_v = y_test[idx[n_test:]]

    # Estimate the threshold on the validation set
    res = _estimate_threshold_metrics(score_v, y_v, pos_label, nq)
    threshold = res["Thresh_star"]

    # Compute metrics on the test set
    metrics = compute_metrics(score_t, y_t, threshold, pos_label)

    return {
        "Precision": metrics[1],
        "Recall": metrics[2],
        "F1-Score": metrics[3],
        "AUPR": metrics[5],
        "AUROC": metrics[4],
        "Thresh_star": threshold,
        # "Quantile_star": qis
    }


def compute_metrics(test_score, y_test, thresh, pos_label=1):
    """
    This function compute metrics for a given threshold

    Parameters
    ----------
    test_score
    y_test
    thresh
    pos_label

    Returns
    -------

    """
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score, max_fpr=1e-2)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm


def _estimate_threshold_metrics(test_score, y_test, pos_label=1, nq=100, optimal=True):
    ratio = 100 * sum(y_test != pos_label) / len(y_test)

    if not optimal:
        thresh = np.percentile(test_score, ratio)
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        return accuracy, precision, recall, f_score, roc, avgpr

    print(f"Ratio of normal data:{round(ratio,2)}%")
    q = np.linspace(max(ratio - 5, .1), min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # Prediction using the threshold value
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }
