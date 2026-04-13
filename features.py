"""特征工程模块 — 结构化特征构建（路径一：LightGBM 输入）"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_merge(path: str) -> pd.DataFrame:
    """加载并合并 transaction 与 identity 表"""
    trans = pd.read_csv(f"{path}/train_transaction.csv")
    iden = pd.read_csv(f"{path}/train_identity.csv")
    return trans.merge(iden, on="TransactionID", how="left")


def load_test_and_merge(path: str) -> pd.DataFrame:
    trans = pd.read_csv(f"{path}/test_transaction.csv")
    iden = pd.read_csv(f"{path}/test_identity.csv")
    return trans.merge(iden, on="TransactionID", how="left")


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """降低 DataFrame 内存占用（float64→float32, int64→int32）"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == np.float64:
            df[col] = df[col].astype(np.float32)
        elif col_type == np.int64:
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    """时间特征分解"""
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["weekday"] = (df["TransactionDT"] // 86400) % 7
    df["is_month_end"] = ((df["TransactionDT"] // 86400) % 30 >= 27).astype(np.int8)
    return df


def amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """金额变换特征"""
    df["amt_log"] = np.log1p(df["TransactionAmt"])
    df["amt_bin"] = pd.qcut(
        df["TransactionAmt"], q=10, labels=False, duplicates="drop"
    )
    return df


def aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """聚合统计特征（card1 维度）"""
    for col in ["card1", "addr1", "P_emaildomain"]:
        grp = df.groupby(col)["TransactionAmt"]
        df[f"{col}_amt_mean"] = df[col].map(grp.mean())
        df[f"{col}_amt_std"] = df[col].map(grp.std())
        df[f"{col}_txn_cnt"] = df[col].map(df.groupby(col).size())
    return df


def device_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """设备指纹哈希"""
    df["device_hash"] = (
        df["DeviceInfo"].astype(str) + "|" + df["id_31"].astype(str)
    ).apply(lambda x: hash(x) % 1_000_000)
    return df


def missing_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """M1-M9 缺失值模式编码"""
    m_cols = [c for c in df.columns if c.startswith("M")]
    df["M_null_count"] = df[m_cols].isnull().sum(axis=1)
    df["M_pattern"] = (
        df[m_cols].isnull().astype(int).astype(str).agg("".join, axis=1)
    )
    df["M_pattern_hash"] = df["M_pattern"].apply(lambda x: hash(x) % 100_000)
    df.drop(columns=["M_pattern"], inplace=True)
    return df


def target_encoding(
    df: pd.DataFrame, col: str, target: str = "isFraud", n_folds: int = 5
) -> pd.DataFrame:
    """5-fold Target Encoding（防止数据泄露）"""
    from sklearn.model_selection import KFold

    global_mean = df[target].mean()
    te_col = f"{col}_te"
    df[te_col] = np.nan

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        df.loc[df.index[val_idx], te_col] = df.iloc[val_idx][col].map(means)

    df[te_col].fillna(global_mean, inplace=True)
    return df


def label_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """对类别特征进行 LabelEncoding"""
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """完整特征工程 pipeline"""
    df = reduce_mem_usage(df)
    df = time_features(df)
    df = amount_features(df)
    df = aggregation_features(df)
    df = device_fingerprint(df)
    df = missing_pattern(df)

    if is_train:
        for col in ["card1", "addr1", "P_emaildomain"]:
            df = target_encoding(df, col)

    df = label_encode_categoricals(df)
    return df
