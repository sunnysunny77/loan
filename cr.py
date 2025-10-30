#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import copy
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, fbeta_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import HistGradientBoostingClassifier

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Constants
lr = 5e-4
weight_decay = 1e-5
batch_size = 64
num_epochs = 75
num_runs = 5
max_patience = 13

# pd 
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# In[56]:


def load_datasets(base_path="./"):

    files = {"train": "cs-training.csv"}
    dfs = {}

    for key, filename in files.items():
        print(f"Loading {filename}...")
        dfs[key] = pd.read_csv(base_path + filename, index_col=0)
        print(f"Loaded {filename} with {len(dfs[key].columns)} columns")

    return dfs

def dataset_summary(df, show_counts=True):

    total_rows = len(df)
    total_duplicates = df.duplicated().sum()

    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null_count": df.notna().sum(),
        "missing_count": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(),
        "duplicates_in_dataset": total_duplicates
    })

    print(f"Dataset shape: {df.shape}")
    if show_counts:
        print(f"Total rows: {total_rows}")
        print(f"Total duplicate rows: {total_duplicates}")
    summary = summary.sort_values(by="missing_%", ascending=False)

    return summary

def outlier_handling(df, target_col, threshold_high=99, threshold_low=1):

    df_copy = df.copy()

    numeric_cols = df_copy.select_dtypes(include=["float64", "int64", "float32"]).columns.tolist()
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0)

    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]

    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42,
        min_samples_leaf=20
    )
    hgb.fit(X, y)

    y_pred_proba = hgb.predict_proba(X)[:, 1]

    high_val = np.percentile(y_pred_proba, threshold_high)
    low_val = np.percentile(y_pred_proba, threshold_low)

    mask = (y_pred_proba < high_val) & (y_pred_proba > low_val)
    df_filtered = df_copy.loc[mask].reset_index(drop=True)

    print(f"Removed {len(df_copy) - len(df_filtered)} extreme rows according to model probabilities")
    print(df_filtered.describe())

    return df_filtered

def drop_target_and_ids(df):

    df_copy = df.copy()
    feature_cols_to_drop = ["SeriousDlqin2yrs"]
    target = df_copy["SeriousDlqin2yrs"]
    df_raw_features = df_copy.drop(columns=feature_cols_to_drop)

    print(f"Dropped target column: {feature_cols_to_drop}")

    return df_raw_features, target, feature_cols_to_drop

def engineer_features(df):
    """
    Create new engineered features for credit risk modeling.

    The goal is to extract behavioral, ratio-based, and interaction features
    from credit-related variables to better capture relationships
    linked to default risk or repayment probability. Generated
    """

    df_engi = df.copy()

    # -------------------------------------------------------------------------
    # AGE CAPPING
    # Cap age at 100 to prevent outliers (e.g., data errors like 200+ years old)
    # and ensure all downstream ratios remain stable.
    # Keep both the capped 'age' and a duplicate 'age_capped' for reference.
    # -------------------------------------------------------------------------
    df_engi["age"] = np.minimum(df_engi["age"], 100)
    df_engi["age_capped"] = df_engi["age"]

    # -------------------------------------------------------------------------
    # DELINQUENCY-RELATED FEATURES
    # Capture overall and major delinquency behavior as indicators of risk.
    # -------------------------------------------------------------------------
    df_engi["TotalPastDue"] = (
        df_engi["NumberOfTime30-59DaysPastDueNotWorse"] +
        df_engi["NumberOfTimes90DaysLate"] +
        df_engi["NumberOfTime60-89DaysPastDueNotWorse"]
    )  # ➤ total count of all delinquent events
    df_engi["HasDelinquencyBinary"] = (df_engi["TotalPastDue"] > 0).astype(int)  # ➤ any delinquency
    df_engi["MajorDelinquencyBinary"] = (df_engi["NumberOfTimes90DaysLate"] > 0).astype(int)  # ➤ serious delinquency

    # -------------------------------------------------------------------------
    # INCOME / DEBT RELATIONSHIPS
    # Key ratios that show how income interacts with debt exposure and credit usage.
    # -------------------------------------------------------------------------
    df_engi["MonthlyDebtAmount"] = df_engi["DebtRatio"] * df_engi["MonthlyIncome"]
    # ➤ estimated monthly debt payment amount

    df_engi["AvailableCreditRatio"] = (
        df_engi["NumberOfOpenCreditLinesAndLoans"] /
        df_engi["NumberRealEstateLoansOrLines"].replace(0, np.nan)
    )
    # ➤ ratio of open credit lines to real estate loans — proxy for portfolio mix

    df_engi["Log_MonthlyIncome"] = np.log1p(df_engi["MonthlyIncome"])
    # ➤ stabilizes skew in income distribution

    df_engi["UtilToAgeRatio"] = (
        df_engi["RevolvingUtilizationOfUnsecuredLines"] /
        df_engi["age"].replace(0, np.nan)
    )
    # ➤ normalizes utilization by age (younger borrowers tend to have higher utilization)

    # -------------------------------------------------------------------------
    # CREDIT MIX FEATURE
    # Encodes the composition of credit exposure to distinguish between borrowers
    # with real-estate vs consumer debt.
    # -------------------------------------------------------------------------
    def credit_mix(row):
        if row["NumberRealEstateLoansOrLines"] == 0 and row["NumberOfOpenCreditLinesAndLoans"] == 0:
            return "NoCredit"
        elif row["NumberRealEstateLoansOrLines"] > 0 and row["NumberOfOpenCreditLinesAndLoans"] == 0:
            return "RealEstateOnly"
        elif row["NumberRealEstateLoansOrLines"] == 0 and row["NumberOfOpenCreditLinesAndLoans"] > 0:
            return "OtherCreditOnly"
        else:
            return "MixedCredit"

    df_engi["CreditMix"] = df_engi.apply(credit_mix, axis=1)

    # -------------------------------------------------------------------------
    # HIGH UTILIZATION FLAG
    # Simple binary threshold — high utilization is a strong risk signal.
    # -------------------------------------------------------------------------
    threshold = 0.8
    df_engi["IsHighUtilizationBinary"] = (
        df_engi["RevolvingUtilizationOfUnsecuredLines"] > threshold
    ).astype(int)

    # -------------------------------------------------------------------------
    # RATIO AND INTERACTION FEATURES
    # Combine multiple behavioral and financial indicators to enrich the signal.
    # -------------------------------------------------------------------------
    df_engi["DebtRatioPerAge"] = (
        df_engi["DebtRatio"] / df_engi["age"].replace(0, np.nan)
    )  # ➤ younger borrowers with high debt ratio = higher risk

    df_engi["MonthlyDebtPerIncome"] = (
        df_engi["MonthlyDebtAmount"] / df_engi["MonthlyIncome"].replace(0, np.nan)
    )  # ➤ fraction of income spent on debt payments

    df_engi["PastDuePerCreditLine"] = (
        df_engi["TotalPastDue"] /
        (df_engi["NumberOfOpenCreditLinesAndLoans"] + df_engi["NumberRealEstateLoansOrLines"]).replace(0, np.nan)
    )  # ➤ delinquency rate per credit line

    df_engi["UtilTimesDebtRatio"] = (
        df_engi["RevolvingUtilizationOfUnsecuredLines"] * df_engi["DebtRatio"]
    )  # ➤ compound exposure to both utilization and indebtedness

    df_engi["AgeTimesIncome"] = df_engi["age"] * df_engi["MonthlyIncome"]
    # ➤ proxy for lifetime earning capacity

    # -------------------------------------------------------------------------
    # AGE BINNING
    # Convert age into ordered categorical buckets for non-linear patterns.
    # -------------------------------------------------------------------------
    df_engi["AgeBin"] = pd.cut(
        df_engi["age"],
        bins=[0, 25, 35, 45, 55, 65, 75, 85, 100],
        labels=[
            "Age_0_25", "Age_25_35", "Age_35_45", "Age_45_55",
            "Age_55_65", "Age_65_75", "Age_75_85", "Age_85_100"
        ],
        include_lowest=True
    )

    # -------------------------------------------------------------------------
    # ADVANCED INTERACTIONS
    # Multiply complementary variables to capture second-order relationships.
    # -------------------------------------------------------------------------
    df_engi["DebtTimesUtil"] = (
        df_engi["DebtRatio"] * df_engi["RevolvingUtilizationOfUnsecuredLines"]
    )  # ➤ joint measure of overall financial stress

    df_engi["IncomePerOpenCredit"] = (
        df_engi["MonthlyIncome"] /
        df_engi["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan)
    )  # ➤ average income per credit line — proxy for capacity

    df_engi["PastDuePerAge"] = (
        df_engi["TotalPastDue"] / df_engi["age"].replace(0, np.nan)
    )  # ➤ delinquency normalized by age

    df_engi["DebtPerCreditTimesAge"] = (
        df_engi["DebtRatioPerAge"] * df_engi["NumberOfOpenCreditLinesAndLoans"]
    )  # ➤ how exposure scales with both debt ratio and age

    # -------------------------------------------------------------------------
    print("Engineered features")

    return df_engi

def drop_high_missing_cols(df, threshold=0.3):

    missing_frac = df.isna().mean()
    hm_cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()

    if hm_cols_to_drop:
        df_drop = df.drop(columns=hm_cols_to_drop)
        print(f"Dropped {len(hm_cols_to_drop)} columns with missing >{threshold*100:.0f}%")
        print(f"Columns dropped: {hm_cols_to_drop}")
    else:
        df_drop = df.copy()
        print("No columns dropped for missing threshold")

    return df_drop, hm_cols_to_drop

def drop_high_card_cols(df, threshold=50):

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    hc_cols_to_drop = [col for col in cat_cols if df[col].nunique() > threshold]

    if hc_cols_to_drop:
        df_high = df.drop(columns=hc_cols_to_drop, errors='ignore')
        print(f"Dropped {len(hc_cols_to_drop)} high-cardinality columns (> {threshold} unique)")
        print(f"Columns dropped: {hc_cols_to_drop}")
    else:
        df_high = df.copy()
        print("No high-cardinality columns dropped")

    return df_high, hc_cols_to_drop

def drop_low_correlated_to_target(df, y, threshold=0.1, drop_direction=None):

    df_temp = df.copy()

    cat_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype('category').cat.codes

    df_temp['__target__'] = y
    corr_with_target = df_temp.corr()['__target__'].drop('__target__')

    if drop_direction is None:
        dropped_cols = corr_with_target[abs(corr_with_target) < threshold].index.tolist()
    elif drop_direction is True:
        dropped_cols = corr_with_target[(corr_with_target > 0) & (corr_with_target < threshold)].index.tolist()
    else: 
        dropped_cols = corr_with_target[(corr_with_target < 0) & (corr_with_target > -threshold)].index.tolist()

    plt.figure(figsize=(10, 6))
    corr_with_target.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'+{threshold}')
    plt.axhline(y=-threshold, color='red', linestyle='--', label=f'-{threshold}')
    plt.title("Feature Correlation with Target")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if dropped_cols:
        df_corr = df.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} features based on correlation and drop_direction={drop_direction}")
        print(f"Columns dropped: {dropped_cols}")
    else:
        df_corr = df.copy()
        print("No features dropped.")

    return df_corr, dropped_cols

def collapse_rare_categories(df, threshold=0.005):

    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    rare_maps = {}
    for col in cat_cols:
        freqs = df_copy[col].value_counts(normalize=True)
        rare_cats = freqs[freqs < threshold].index
        if len(rare_cats) > 0:
            df_copy[col] = df_copy[col].astype('object').replace(rare_cats, 'Other')
            rare_maps[col] = set(rare_cats)
            print(f"Collapsed {len(rare_cats)} rare categories in column '{col}'")

            print(f"Categories dropped: {list(rare_cats)}")
    if not rare_maps:
        print("No rare categories collapsed")
        rare_maps = None

    return df_copy, rare_maps

def impute_and_scale(df, threshold=1.0):

    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    num_imputer = None
    cat_imputer = None
    robust_scaler = None
    std_scaler = None

    if numeric_cols:
        df_copy[numeric_cols] = df_copy[numeric_cols].replace([np.inf, -np.inf], np.nan)

        for col in numeric_cols:
            df_copy[f'Was{col}Imputed'] = df_copy[col].isna().astype(int)

        num_imputer = SimpleImputer(strategy='median')
        df_copy[numeric_cols] = num_imputer.fit_transform(df_copy[numeric_cols])

    if cat_cols:
        for col in cat_cols:
            df_copy[f'Was{col}Imputed'] = df_copy[col].isna().astype(int)

        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_copy[cat_cols] = cat_imputer.fit_transform(df_copy[cat_cols])

    if numeric_cols:
        skewness = pd.DataFrame(df_copy[numeric_cols]).skew().sort_values(ascending=False)
        skewed_cols = skewness[abs(skewness) > threshold].index.tolist()

        if skewed_cols:
            robust_scaler = RobustScaler()
            df_copy[skewed_cols] = robust_scaler.fit_transform(df_copy[skewed_cols]).astype(np.float32)

        normal_cols = [c for c in numeric_cols if c not in skewed_cols]
        if normal_cols:
            std_scaler = StandardScaler()
            df_copy[normal_cols] = std_scaler.fit_transform(df_copy[normal_cols]).astype(np.float32)

    df_processed = df_copy.copy()

    print("Imputed, flagged, and scaled features")

    return df_processed, num_imputer, cat_imputer, robust_scaler, std_scaler

def select_features_xgb(df, target, threshold=None, random_state=42):

    df_temp = df.copy()

    imputed_flag_cols = [col for col in df_temp.columns if re.match(r'^Was.+Imputed$', col)]
    if imputed_flag_cols:
        df_temp = df_temp.drop(columns=imputed_flag_cols, errors='ignore')

    cat_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype('category').cat.codes

    X_train, X_val, y_train, y_val = train_test_split(
        df_temp, target, test_size=0.2, random_state=random_state, stratify=target
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 5,
        "min_child_weight": 2,
        "gamma": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eta": 0.05,
        "lambda": 1,
        "alpha": 0,
        "tree_method": "hist",
        "max_delta_step": 1
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    importance_dict = model.get_score(importance_type='gain')
    all_importances = pd.Series(0, index=df_temp.columns, dtype=float)

    for k, v in importance_dict.items():
        all_importances[k] = v
    all_importances = all_importances.sort_values(ascending=False)

    if threshold is None:
        selected_features = all_importances[all_importances > 0].index.tolist()
        print(f"Using gain > 0: selected {len(selected_features)} features")
    else:
        selected_features = all_importances[all_importances >= threshold].index.tolist()
        print(f"Using gain ≥ {threshold}: selected {len(selected_features)} features")

    dropped_features = [col for col in df_temp.columns if col not in selected_features]

    if dropped_features:
        print(f"Dropped {len(dropped_features)} features")
        print(f"Columns dropped: {dropped_features}")
    else:
        print("No features dropped")

    imputed_flag_caps = [col.upper() for col in imputed_flag_cols if col in df.columns]
    final_features = selected_features + imputed_flag_caps
    df_selected = df.reindex(columns=final_features, fill_value=0).copy()

    plt.figure(figsize=(10, 6))
    plt.barh(all_importances.index[::-1], all_importances.values[::-1], color='skyblue')
    plt.xlabel("Feature Importance (gain)")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

    return df_selected, final_features

def transform_val_test(df, cols_to_drop, selected_features, rare_maps, num_imputer, cat_imputer, robust_scaler, std_scaler):

    df_copy = df.copy()

    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop, errors='ignore')

    if rare_maps:
        for col, rare_cats in rare_maps.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: x if x not in rare_cats else 'Other')

    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()

    if numeric_cols and num_imputer:
        df_copy[numeric_cols] = df_copy[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_copy[numeric_cols] = num_imputer.transform(df_copy[numeric_cols])

    if robust_scaler:
        skewed_cols = robust_scaler.feature_names_in_
        df_copy[skewed_cols] = robust_scaler.transform(df_copy[skewed_cols]).astype(np.float32)

    if std_scaler:
        normal_cols = [c for c in numeric_cols if robust_scaler is None or c not in robust_scaler.feature_names_in_]
        if normal_cols:
            df_copy[normal_cols] = std_scaler.transform(df_copy[normal_cols]).astype(np.float32)

    cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols and cat_imputer:
        df_copy[cat_cols] = cat_imputer.transform(df_copy[cat_cols])

    if selected_features:
        df_copy = df_copy.reindex(columns=selected_features, fill_value=0)

    return df_copy

def check_and_drop_duplicates(df, target=None, drop_target_na=False, show_info=True):

    df_cleaned = df.copy()
    target_cleaned = None

    total_duplicates = df_cleaned.duplicated().sum()
    if total_duplicates > 0:
        df_cleaned = df_cleaned.drop_duplicates(keep='first')
        if show_info:
            print(f"Dropped {total_duplicates} duplicate rows. Remaining: {len(df_cleaned)}")

    if target is not None:
        target_cleaned = pd.Series(target).reindex(df_cleaned.index)
        if drop_target_na:
            mask = target_cleaned.notna()
            dropped = len(target_cleaned) - mask.sum()
            if dropped > 0 and show_info:
                print(f"Dropped {dropped} rows with missing target values")
            df_cleaned = df_cleaned.loc[mask].reset_index(drop=True)
            target_cleaned = target_cleaned.loc[mask].reset_index(drop=True)
        else:
            target_cleaned = target_cleaned.reset_index(drop=True)
        return df_cleaned, target_cleaned
    else:
        return df_cleaned


# In[57]:


# Load datasets
dfs = load_datasets()
df_train = dfs["train"]


# In[58]:


#summary
print(dataset_summary(df_train))
print(df_train.head(5))


# In[59]:


# Outlier Handling
df_train = df_train[df_train['age'] > 0].reset_index(drop=True)
df_filtered = outlier_handling(df_train, target_col="SeriousDlqin2yrs", threshold_high=99.99, threshold_low=0.01)


# In[60]:


# Select targets
df_features, target, feature_cols_to_drop = drop_target_and_ids(df_filtered)
print(target.value_counts())


# In[61]:


# Split train/test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    df_features, target, test_size=0.2, stratify=target, random_state=42
)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)


# In[62]:


# Engineer_features
df_engi = engineer_features(X_train)


# In[67]:


# Drop columns with missing
df_drop, hm_cols_to_drop = drop_high_missing_cols(df_engi, threshold=0.30)


# In[68]:


# Drop high card
df_high, hc_cols_to_drop = drop_high_card_cols(df_drop, threshold=50)


# In[69]:


# Drop low correlated features to target
df_corr, low_corr_cols_to_drop = drop_low_correlated_to_target(df_high, y_train, threshold=0.0067, drop_direction=False)


# In[70]:


# Collapse rare categories
df_collapsed, rare_maps = collapse_rare_categories(df_corr, threshold=0.05)


# In[71]:


# Impute and scale
df_processed, num_imputer, cat_imputer, robust_scaler, std_scaler  = impute_and_scale(df_collapsed , threshold=1.0)


# In[72]:


# Feature selection
df_selected, selected_features = select_features_xgb(df_processed, y_train, threshold=None)


# In[73]:


# Process
all_cols_to_drop = feature_cols_to_drop + hm_cols_to_drop + low_corr_cols_to_drop + hc_cols_to_drop

X_val = engineer_features(X_val)
X_val = transform_val_test(X_val, all_cols_to_drop, selected_features, rare_maps, num_imputer, cat_imputer, robust_scaler, std_scaler)

X_test = engineer_features(X_test)
X_test = transform_val_test(X_test, all_cols_to_drop, selected_features, rare_maps, num_imputer, cat_imputer, robust_scaler, std_scaler)
X_train = df_selected.copy()


# In[74]:


# Drop duplicates
X_train, y_train = check_and_drop_duplicates(X_train, y_train)


# In[75]:


#summary
print(dataset_summary(X_train))


# In[76]:


# Encode
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

cat_maps = {
    col: {cat: idx for idx, cat in enumerate(X_train[col].astype(str).unique())}
    for col in cat_cols
}

for col in cat_cols:
    X_train[col] = X_train[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)
    X_val[col] = X_val[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)
    X_test[col] = X_test[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)

num_cols = [col for col in X_train.columns if col not in cat_cols]

X_train_num = X_train[num_cols].astype('float32').values
X_val_num = X_val[num_cols].astype('float32').values
X_test_num = X_test[num_cols].astype('float32').values

X_train_cat = X_train[cat_cols].astype('int64').values
X_val_cat = X_val[cat_cols].astype('int64').values
X_test_cat = X_test[cat_cols].astype('int64').values


# In[77]:


# Convert to tensors
X_train_num_tensor = torch.tensor(X_train_num)
X_val_num_tensor = torch.tensor(X_val_num)
X_test_num_tensor = torch.tensor(X_test_num)

X_train_cat_tensor = torch.tensor(X_train_cat)
X_val_cat_tensor = torch.tensor(X_val_cat)
X_test_cat_tensor = torch.tensor(X_test_cat)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
weights_tensor = torch.tensor([class_weight_dict[int(c)] for c in y_train], dtype=torch.float32)

print("Numeric input shape:", X_train_num_tensor.shape)
print("Categorical input shape:", X_train_cat_tensor.shape)
print("Class weights:", class_weight_dict)


# In[78]:


# Datasets
class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]

train_ds = TabularDataset(X_train_num_tensor, X_train_cat_tensor, y_train_tensor)
val_ds = TabularDataset(X_val_num_tensor, X_val_cat_tensor, y_val_tensor)
test_ds = TabularDataset(X_test_num_tensor, X_test_cat_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")


# In[79]:


# Model
class NN(nn.Module):
    def __init__(self, num_numeric, cat_dims, emb_dims):
        super().__init__()

        self.emb_layers = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim)
            for cat_dim, emb_dim in zip(cat_dims, emb_dims)
        ])
        self.emb_dropout = nn.Dropout(0.3)

        self.bn_num = nn.BatchNorm1d(num_numeric)

        total_emb_dim = sum(emb_dims)
        self.input_dim = num_numeric + total_emb_dim

        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(144, 72),
            nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.skip_proj_main = nn.Sequential(
            nn.Linear(self.input_dim, 72),
            nn.Dropout(0.3)
        )

        self.cat_skip = nn.Sequential(
            nn.Linear(total_emb_dim, 72),
            nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.out = nn.Linear(72, 1)

    def forward(self, x_num, x_cat):

        x_cat_emb = torch.cat([
            emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)
        ], dim=1)
        x_cat_emb = self.emb_dropout(x_cat_emb)

        x_num = self.bn_num(x_num)

        x = torch.cat([x_num, x_cat_emb], dim=1)

        x_main = self.main(x)

        x_skip = self.skip_proj_main(x) + self.cat_skip(x_cat_emb)

        x_combined = x_main + x_skip
        return self.out(x_combined).squeeze(1)

cat_dims = [len(cat_maps[col]) for col in cat_cols]
emb_dims = [min(50, (cat_dim + 1) // 2) for cat_dim in cat_dims]

model = NN(X_train_num.shape[1], cat_dims, emb_dims).to(device)
print(model)
print("Total parameters:", sum(p.numel() for p in model.parameters()))


# In[80]:


# Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device)
            if self.pos_weight else None
        )
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


alpha = class_weights[1] / (class_weights[0] + class_weights[1])
loss_fn = FocalLoss(alpha=alpha, gamma=3)


# In[81]:


# Train
overall_best_val_auc = 0.0
overall_best_model_state = None

for run in range(num_runs):
    print(f"\n=== Run {run + 1}/{num_runs} ===")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_val_auc_this_run = 0.0
    best_model_state_this_run = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_logits, train_labels = [], []

        for x_num, x_cat, yb in train_loader:
            x_num, x_cat, yb = x_num.to(device), x_cat.to(device), yb.to(device).float()

            optimizer.zero_grad()
            logits = model(x_num, x_cat)  
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_num.size(0)
            train_logits.append(logits.detach().cpu())
            train_labels.append(yb.cpu())

        train_loss = total_train_loss / len(train_loader.dataset)
        train_logits = torch.cat(train_logits)
        train_labels = torch.cat(train_labels)
        train_probs = torch.sigmoid(train_logits).numpy()
        train_auc = roc_auc_score(train_labels.numpy(), train_probs)

        model.eval()
        total_val_loss = 0.0
        val_logits, val_labels = [], []

        with torch.no_grad():
            for x_num, x_cat, yb in val_loader:
                x_num, x_cat, yb = x_num.to(device), x_cat.to(device), yb.to(device).float()
                logits = model(x_num, x_cat)

                loss = loss_fn(logits, yb)
                total_val_loss += loss.item() * x_num.size(0)
                val_logits.append(logits.cpu())
                val_labels.append(yb.cpu())

        val_loss = total_val_loss / len(val_loader.dataset)
        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(val_labels.numpy(), val_probs)

        if val_auc > best_val_auc_this_run:
            best_val_auc_this_run = val_auc
            best_model_state_this_run = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_auc)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {train_loss:.6f} | Train AUC: {train_auc:.4f} | "
              f"Val loss: {val_loss:.6f} | Val AUC: {val_auc:.4f}")

    print(f"Run {run + 1} best Val AUC: {best_val_auc_this_run:.4f}")

    if best_val_auc_this_run > overall_best_val_auc:
        overall_best_val_auc = best_val_auc_this_run
        overall_best_model_state = copy.deepcopy(best_model_state_this_run)

model.load_state_dict(overall_best_model_state)
print(f"\nBest model across all runs restored (Val AUC = {overall_best_val_auc:.4f})")


# In[82]:


# Evaluation
model.eval()
y_val_probs = []

with torch.no_grad():
    for x_num, x_cat, yb in val_loader:
        x_num, x_cat = x_num.to(device), x_cat.to(device)
        outputs = model(x_num, x_cat)
        probs = torch.sigmoid(outputs)
        y_val_probs.extend(probs.cpu().numpy())

y_val_probs = np.array(y_val_probs)

# Target defaults recall
prec, rec, thresholds = precision_recall_curve(y_val, y_val_probs)
f_beta_scores = [fbeta_score(y_val, (y_val_probs > t).astype(int), beta=2) for t in thresholds]
best_thresh = thresholds[np.argmax(f_beta_scores)]

y_test_probs = []
with torch.no_grad():
    for x_num, x_cat, _ in test_loader:
        x_num, x_cat = x_num.to(device), x_cat.to(device)
        outputs = model(x_num, x_cat)
        probs = torch.sigmoid(outputs)
        y_test_probs.extend(probs.cpu().numpy())

y_test_probs = np.array(y_test_probs)
y_test_pred_opt = (y_test_probs > best_thresh).astype(int)

target_names = ['Repaid', 'Defaulted']
report = classification_report(y_test, y_test_pred_opt, target_names=target_names)
acc = accuracy_score(y_test, y_test_pred_opt)
roc_auc = roc_auc_score(y_test, y_test_probs)
cm = confusion_matrix(y_test, y_test_pred_opt)
tn, fp, fn, tp = cm.ravel()
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print("Best threshold for F1:", best_thresh)
print(report)
print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC AUC: {roc_auc:.3f}")
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
for i, class_name in enumerate(target_names):
    print(f"Accuracy for class '{class_name}': {per_class_acc[i]*100:.2f}%")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold = {best_thresh:.2f})")
plt.show()


# In[83]:


# Data sets
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test) 


# In[84]:


# Model
neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
scale_pos_weight = neg_count / pos_count
print("scale_pos_weight:", scale_pos_weight)

params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "scale_pos_weight": scale_pos_weight,
    "max_depth": 5,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eta": 0.05,
    "gamma": 1,
    "lambda": 1,
    "alpha": 0
}

evals = [(dtrain, "train"), (dval, "validation")]


# In[85]:


# Train
model_b = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)


# In[86]:


# Evaluation
y_probs = model_b.predict(dtest) 

# Target defaults recall
prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
f_beta_scores = [fbeta_score(y_test, (y_probs > t).astype(int), beta=2) for t in thresholds]
best_thresh = thresholds[np.argmax(f_beta_scores)]

y_pred = (y_probs > best_thresh).astype(int)

target_names = ['Repaid', 'Defaulted']
report = classification_report(y_test, y_pred, target_names=target_names)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
per_class_acc = cm.diagonal() / cm.sum(axis=1)
roc_auc = roc_auc_score(y_test, y_probs)

print("Best threshold for F1:", best_thresh)
print(report)
print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC AUC: {roc_auc:.3f}")
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
for i, class_name in enumerate(target_names):
    print(f"Accuracy for class '{class_name}': {per_class_acc[i]*100:.2f}%")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold = {best_thresh:.2f})")
plt.show()


# In[ ]:




