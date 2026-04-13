"""LightGBM 路径 — 结构化特征分类器"""

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


def train_kfold(
    X, y, params=None, n_folds: int = 5, early_stopping_rounds: int = 100
):
    """
    K-fold 交叉验证训练 LightGBM。

    Returns: (models, oof_predictions, mean_auc)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(200),
            ],
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}")
        models.append(model)

    mean_auc = roc_auc_score(y, oof_preds)
    print(f"\n  Overall OOF AUC: {mean_auc:.5f}")
    return models, oof_preds, mean_auc
