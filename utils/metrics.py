from typing import Optional
import numpy as np


def abs_rel_diff(y_true: np.ndarray, y_pred: np.ndarray):
    """Absolute relative difference."""
    return np.mean(np.abs(y_pred - y_true) / y_true)


def squa_rel_diff(y_true: np.ndarray, y_pred: np.ndarray):
    """Squared relative difference."""
    return np.mean(np.square(y_pred - y_true) / y_true)


def rmse_linear(y_true: np.ndarray, y_pred: np.ndarray):
    """Root mean squared error."""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray):
    """Root mean squared error."""
    EPS = np.finfo(y_true.dtype).eps
    y_pred = np.clip(y_pred, EPS, None)
    return np.sqrt(np.mean(np.square(np.log(y_pred) - np.log(y_true + EPS))))


def threshold_accuracy(y_true: np.ndarray, y_pred: np.ndarray, th: float = 1.25):
    """Threshold accuracy."""
    threshold = np.maximum((y_true / y_pred), (y_pred / y_true))
    return np.mean(threshold < th)


def rel_dist_error(y_true: np.ndarray, y_pred: np.ndarray, th: float = 0.05):
    # calculate relative distance errors for each object and then percentage below threshold
    dist_errors = abs((y_true - y_pred) / y_true)

    return float(len(dist_errors[dist_errors < th])) / len(y_true) * 100


def ale(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Average localization error."""
    th = [0, 10, 20, 30, 100]
    dist = np.abs(y_true - y_pred)
    ale_ = {f"ale_{th1}-{th2}": np.mean(dist[(y_true >= th1) & (y_true < th2)]) for th1, th2 in zip(th[:-1], th[1:])}
    ale_all = {"ale_all": np.mean(dist)}
    return ale_ | ale_all


def alp(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Average localization precision."""
    th = [0.5, 1, 2]
    dist = np.abs(y_true - y_pred)
    return {f"alp_@{th1}m": (len(dist[dist < th1]) / len(dist)) if len(dist) != 0 else None for th1 in th}


def aloe(y_true: np.ndarray, y_pred: np.ndarray, y_occlusions: np.ndarray) -> dict:
    """Average localization of occluded objects error."""
    th = [0.0, 0.3, 0.5, 0.75, 1.0]
    dist = np.abs(y_true - y_pred)
    aloe_ = {f"aloe_{th1}-{th2}": np.mean(dist[(y_occlusions >= th1) & (y_occlusions < th2)]) for th1, th2 in
             zip(th[:-1], th[1:])}
    aloe_all = {"aloe_all": np.mean(dist)}
    return aloe_ | aloe_all


def get_metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray, y_visibilities: Optional[list] = None) -> dict:
    ale_ = ale(y_true, y_pred)
    alp_ = alp(y_true, y_pred)
    if y_visibilities is not None:
        y_occlusions = 1 - np.array(y_visibilities)
        aloe_ = aloe(y_true, y_pred, y_occlusions)
    else:
        aloe_ = {}

    return {
               "abs_rel_diff": abs_rel_diff(y_true, y_pred),
               "squa_rel_diff": squa_rel_diff(y_true, y_pred),
               "rmse_linear": rmse_linear(y_true, y_pred),
               "rmse_log": rmse_log(y_true, y_pred),
               "delta_1": threshold_accuracy(y_true, y_pred),
               "delta_2": threshold_accuracy(y_true, y_pred, th=1.25 ** 2),
               "delta_3": threshold_accuracy(y_true, y_pred, th=1.25 ** 3),
               "rel_dist_5": rel_dist_error(y_true, y_pred, th=0.05),
               "rel_dist_10": rel_dist_error(y_true, y_pred, th=0.1),
               "rel_dist_15": rel_dist_error(y_true, y_pred, th=0.15),
           } | ale_ | alp_ | aloe_


def get_metrics(y_true: list, y_pred: list, y_visibilities: Optional[list] = None, y_classes: Optional[list] = None,
                long_range=True) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_visibilities = np.array(y_visibilities) if y_visibilities is not None else None
    y_classes = np.array(y_classes) if y_classes is not None else None
    results = {'all': get_metrics_per_class(y_true, y_pred, y_visibilities)}

    if y_classes is not None:
        kitti_classes = {0: 'car', 1: 'van', 2: 'truck', 3: 'pedestrian', 4: 'p_sitting',
                         5: 'cyclist', 6: 'tram'}
        unique_classes = np.unique(y_classes)
        kitti_classes = {cls_id: cls_name for cls_id, cls_name in kitti_classes.items() if cls_id in unique_classes}
        for cls_id, cls_name in kitti_classes.items():
            y_visibilities_cls = y_visibilities[y_classes == cls_id] if y_visibilities is not None else None
            results[cls_name] = get_metrics_per_class(y_true[y_classes == cls_id],
                                                      y_pred[y_classes == cls_id],
                                                      y_visibilities_cls)
            if long_range and cls_name == 'car':
                y_visibilities_long_range = np.array(y_visibilities)[
                    y_true > 40] if y_visibilities is not None else None
                y_classes_long_range = y_classes[y_true > 40]
                y_pred_long_range = y_pred[y_true > 40]
                y_true_long_range = y_true[y_true > 40]
                results['long_range'] = get_metrics_per_class(y_true_long_range[y_classes_long_range == cls_id],
                                                              y_pred_long_range[y_classes_long_range == cls_id],
                                                              y_visibilities_long_range[y_classes_long_range == cls_id])

    return results


PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'


def print_metrics(metrics: dict):
    print(f"{BOLD}Metrics")
    print(f"{'':<18}", ('{:<12}' * len(metrics)).format(*metrics.keys()) + END)
    for metric_name in metrics['all']:
        all_classes_metric = [metrics[cls_id][metric_name] for cls_id in metrics]
        arrow = f"{BOLD}{GREEN} ↑{END}" if metric_name.startswith(("delta", "alp")) else f"{BOLD}{RED} ↓{END}"
        arrow = "" if metric_name.startswith('num') else arrow
        print(f"{metric_name:<15}{arrow} ", ('{:<12.4f}' * len(all_classes_metric)).format(*all_classes_metric))
