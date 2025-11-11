import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Union

def engineer_features(df):
    
    df_e = df.copy()

    NumberOfTime3059DaysPastDueNotWorse = df_e["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0).clip(upper=10)
    NumberOfTimes90DaysLate = df_e["NumberOfTimes90DaysLate"].fillna(0).clip(upper=10)
    NumberOfTime6089DaysPastDueNotWorse = df_e["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0).clip(upper=10)

    TotalPastDue = (
        NumberOfTime3059DaysPastDueNotWorse
        + NumberOfTimes90DaysLate
        + NumberOfTime6089DaysPastDueNotWorse
    )

    RevolvingUtilizationOfUnsecuredLinesCapped = df_e["RevolvingUtilizationOfUnsecuredLines"].clip(upper=5.0).fillna(0.0).replace(0, np.nan)
    RevolvingUtilizationOfUnsecuredLines = np.log1p(RevolvingUtilizationOfUnsecuredLinesCapped)

    AgeSafe = df_e["age"].replace(0, np.nan)

    MonthlyIncomeSafe = df_e["MonthlyIncome"]

    DebtRatioCapped = df_e["DebtRatio"].clip(upper=10000.0)

    CreditLinesSafe = df_e["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan)

    DebtToIncome = DebtRatioCapped * MonthlyIncomeSafe
    IncomePerCreditLine = MonthlyIncomeSafe / CreditLinesSafe

    AgeRisk = np.where(AgeSafe < 25, 1.0,
                 np.where(AgeSafe < 35, 0.8,
                 np.where(AgeSafe < 50, 0.6, 0.4)))

    DelinquencyScore = (
        NumberOfTime3059DaysPastDueNotWorse +
        NumberOfTime6089DaysPastDueNotWorse * 2 +
        NumberOfTimes90DaysLate * 3
    )

    HasAnyDelinquency = (TotalPastDue > 0).astype(int)
    
    df_e["DelinquencyScore"] = DelinquencyScore
    df_e["HasAnyDelinquency"] = HasAnyDelinquency
    df_e["HasMajorDelinquency"] = (
        (NumberOfTime6089DaysPastDueNotWorse > 0) |
        (NumberOfTimes90DaysLate > 0)
    ).astype(int)

    df_e["UtilizationPerAge"] = RevolvingUtilizationOfUnsecuredLines / AgeSafe
    df_e["LatePaymentsPerCreditLine"] = TotalPastDue / CreditLinesSafe

    df_e["IncomePerCreditLine"] = IncomePerCreditLine
    df_e["DebtToIncomeAgeRisk"] = DebtToIncome * AgeRisk
    df_e["AgeRisk"] = AgeRisk

    DelinquencyScore_bins = [-1, 0, 1, 3, 6, np.inf]
    DelinquencyScore_labels = ["None", "Few", "Moderate", "Frequent", "Chronic"]
    df_e["DelinquencyBucket"] = pd.cut(DelinquencyScore, bins=DelinquencyScore_bins, labels=DelinquencyScore_labels)

    Utilization_bins = [-0.01, 0.1, 0.3, 0.6, 0.9, 1.5, 10]
    Utilization_labels = ["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"]
    UtilizationBucket = pd.cut(RevolvingUtilizationOfUnsecuredLines, bins=Utilization_bins, labels=Utilization_labels)

    Late_bins = [-1, 0, 1, 3, 6, np.inf]
    Late_labels = ["NoLate", "FewLate", "ModerateLate", "FrequentLate", "ChronicLate"]
    LatePaymentBucket = pd.cut(TotalPastDue, bins=Late_bins, labels=Late_labels)

    df_e["UtilizationBucketLateBucket"] = (
        UtilizationBucket.astype(str) + "_" + LatePaymentBucket.astype(str)
    )

    engineered_cols = [
        "DelinquencyScore",
        "HasAnyDelinquency",
        "HasMajorDelinquency",
        "UtilizationPerAge",
        "LatePaymentsPerCreditLine",
        "IncomePerCreditLine",
        "DebtToIncomeAgeRisk",
        "AgeRisk",
        "DelinquencyBucket",
        "UtilizationBucketLateBucket"
    ]

    engineered_df = df_e[engineered_cols]

    return engineered_df

class NN(nn.Module):
    def __init__(self, num_numeric, cat_dims, emb_dims):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, emb_dims, strict=True)])
        self.emb_dropout = nn.Dropout(0.3)
        self.bn_num = nn.BatchNorm1d(num_numeric)
        total_emb_dim = sum(emb_dims)
        self.input_dim = num_numeric + total_emb_dim
        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.skip_proj_main = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Dropout(0.3)
        )
        self.cat_skip = nn.Sequential(
            nn.Linear(total_emb_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.out = nn.Linear(64, 1)
    def forward(self, x_num, x_cat):
        x_cat_emb = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)], dim=1)
        x_cat_emb = self.emb_dropout(x_cat_emb)
        x_num = self.bn_num(x_num)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        x_main = self.main(x)
        x_skip = self.skip_proj_main(x) + self.cat_skip(x_cat_emb)
        x_combined = x_main + x_skip
        return self.out(x_combined).squeeze(1)

model_b = xgb.XGBClassifier()
model_b.load_model("cr_b.json")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
robust_scaler = joblib.load("robust_scaler.pkl")
std_scaler = joblib.load("std_scaler.pkl")
cat_maps = joblib.load("cat_maps.pkl")
cat_col_order = joblib.load("cat_col_order.pkl")
X_train_flags = joblib.load("X_train_flags.pkl")
num_col_order = joblib.load("num_col_order.pkl")
skewed_col_order = joblib.load("skewed_col_order.pkl")
threshold_a = joblib.load("threshold_a.pkl")
threshold_b = joblib.load("threshold_b.pkl")
rare_maps = joblib.load("rare_maps.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cat_dims = [len(cat_maps[c]) for c in cat_col_order]
emb_dims = [min(50, (len(cat_maps[c]) + 1) // 2) for c in cat_col_order]

model = NN(num_numeric=(len(num_col_order) + len(X_train_flags)), cat_dims=cat_dims, emb_dims=emb_dims)
weights_path = "cr_weights.pth"
loaded_weights = torch.load(weights_path, map_location=device) 
model.load_state_dict(loaded_weights)
model = model.to(device)
model.eval()

app = FastAPI(title="Credit Risk Prediction API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    data: Dict[str, Union[str, float, int, None]]

def preprocess(df: pd.DataFrame, for_xgb: bool = False):
    
    df_copy = engineer_features(df)

    if num_col_order:     
        df_copy[num_col_order] = df_copy[num_col_order].replace([np.inf, -np.inf], np.nan)
        for col in num_col_order:
            df_copy[f'Was{col}Imputed'] = df_copy[col].isna().astype(int)
        df_copy[num_col_order] = num_imputer.transform(df_copy[num_col_order])
        skewed_cols = skewed_col_order or []
        normal_cols = [c for c in num_col_order if c not in skewed_cols]
        if skewed_cols and robust_scaler:
            df_copy[skewed_cols] = robust_scaler.transform(df_copy[skewed_cols])
        if normal_cols and std_scaler:
            df_copy[normal_cols] = std_scaler.transform(df_copy[normal_cols])
            
    if cat_col_order:
        df_copy[cat_col_order] = df_copy[cat_col_order].astype('object')
        for col in cat_col_order:
            df_copy[f'Was{col}Imputed'] = df_copy[col].isna().astype(int)
        for col in cat_col_order:
            if rare_maps and col in rare_maps:
                rare_categories = list(rare_maps[col])
                df_copy[col] = df_copy[col].replace(rare_categories, 'Other')
        df_copy[cat_col_order] = cat_imputer.transform(df_copy[cat_col_order])

    if for_xgb:
        for col in cat_col_order:
            df_copy[col] = df_copy[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)
        imputation_flags = [f for f in X_train_flags if f in df_copy.columns]   
        df_final = df_copy[num_col_order + imputation_flags + cat_col_order].astype(np.float32)    
        trained_features = model_b.get_booster().feature_names
        df_final = df_final.reindex(columns=trained_features, fill_value=0.0)
        return df_final
    else:
        for col in cat_col_order:
            df_copy[col] = df_copy[col].astype(str).map(cat_maps[col]).astype(int)
        imputation_flags = [f for f in X_train_flags if f in df_copy.columns]
        x_num_tensor = torch.tensor(df_copy[num_col_order + imputation_flags].values, dtype=torch.float32).to(device)
        x_cat_tensor = torch.tensor(df_copy[cat_col_order].values, dtype=torch.int64).to(device)
        return x_num_tensor, x_cat_tensor

def predict_nn(df: pd.DataFrame, threshold=threshold_a):
    x_num, x_cat = preprocess(df, for_xgb=False)
    with torch.no_grad():
        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)
    return probs, preds

@app.post("/predict_nn")
def predict_endpoint(input_data: InputData):
    df = pd.DataFrame([input_data.data]).replace("", np.nan)
    probs, preds = predict_nn(df)
    return {"probabilities": probs.tolist(), "predictions": preds.tolist()}

def predict_xgb(df: pd.DataFrame, threshold=threshold_b):
    df = preprocess(df, for_xgb=True)
    probs = model_b.predict_proba(df)[:, 1]
    preds = (probs > threshold).astype(int)
    return probs, preds

@app.post("/predict_xgb")
def predict_xgb_endpoint(input_data: InputData):
    df = pd.DataFrame([input_data.data]).replace("", np.nan)
    probs, preds = predict_xgb(df)
    return {"probabilities": probs.tolist(), "predictions": preds.tolist()}