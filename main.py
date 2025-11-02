import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Union

def engineer_features(df, expected_num_cols=None, expected_cat_cols=None):

    df_engi = df.copy()
    df_engi = df_engi.fillna(0)
    df_engi["TotalPastDue"] = (
        df_engi["NumberOfTime30-59DaysPastDueNotWorse"]
        + df_engi["NumberOfTimes90DaysLate"]
        + df_engi["NumberOfTime60-89DaysPastDueNotWorse"]
    )  
    df_engi["HasDelinquencyBinary"] = (df_engi["TotalPastDue"] > 0).astype(int)
    df_engi["MajorDelinquencyBinary"] = (
        (df_engi["NumberOfTimes90DaysLate"] > 0)
        | (df_engi["NumberOfTime60-89DaysPastDueNotWorse"] > 0)
    ).astype(int)
    df_engi["IsHighUtilizationBinary"] = (df_engi["RevolvingUtilizationOfUnsecuredLines"] > 0.67).astype(int)
    age_notna = df_engi["age"].notna()
    df_engi.loc[age_notna, "AgeBin"] = pd.cut(
        df_engi.loc[age_notna, "age"],
        bins=[0, 25, 35, 45, 55, 65, 75, 85, 100],
        labels=[
            "Age_0_25", "Age_25_35", "Age_35_45", "Age_45_55",
            "Age_55_65", "Age_65_75", "Age_75_85", "Age_85_100"
        ],
        include_lowest=True
    )
    def credit_mix(row):
        if pd.isna(row["NumberRealEstateLoansOrLines"]) or pd.isna(row["NumberOfOpenCreditLinesAndLoans"]):
            return np.nan

        if row["NumberRealEstateLoansOrLines"] == 0 and row["NumberOfOpenCreditLinesAndLoans"] == 0:
            return "NoCredit"
        elif row["NumberRealEstateLoansOrLines"] > 0 and row["NumberOfOpenCreditLinesAndLoans"] == 0:
            return "RealEstateOnly"
        elif row["NumberRealEstateLoansOrLines"] == 0 and row["NumberOfOpenCreditLinesAndLoans"] > 0:
            return "OtherCreditOnly"
        else:
            return "MixedCredit"
    df_engi["CreditMix"] = df_engi.apply(credit_mix, axis=1)
    df_engi["IsCreditMixRisky"] = df_engi["CreditMix"].ne("MixedCredit").astype(int)
    df_engi["HasDebtRatioHigh"] = (df_engi["DebtRatio"] > 0.67).astype(int)
    df_engi["Has90DaysLate"] = (df_engi["NumberOfTimes90DaysLate"] > 0).astype(int)
    df_engi["HasAnyLate"] = (
        df_engi["NumberOfTimes90DaysLate"]
        + df_engi["NumberOfTime30-59DaysPastDueNotWorse"]
        + df_engi["NumberOfTime60-89DaysPastDueNotWorse"]
        > 0
    ).astype(int)
    df_engi["HasMultipleLate"] = (
        df_engi["NumberOfTimes90DaysLate"]
        + df_engi["NumberOfTime30-59DaysPastDueNotWorse"]
        + df_engi["NumberOfTime60-89DaysPastDueNotWorse"]
        >= 2
    ).astype(int)
    df_engi["HasHighOpenCreditLines"] = (df_engi["NumberOfOpenCreditLinesAndLoans"] > 8).astype(int)
    df_engi["HasHighDebtLoad"] = (
        (df_engi["DebtRatio"] > 0.5) & (df_engi["RevolvingUtilizationOfUnsecuredLines"] > 0.67)
    ).astype(int)
    df_engi["DebtToIncomeRatio"] = df_engi["DebtRatio"] / (df_engi["MonthlyIncome"] + 1e-3)

    return df_engi

class NN(nn.Module):
    def __init__(self, num_numeric, cat_dims, emb_dims):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, emb_dims)])
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

num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
robust_scaler = joblib.load("robust_scaler.pkl")
std_scaler = joblib.load("std_scaler.pkl")
cat_maps = joblib.load("cat_maps.pkl")
cat_col_order = joblib.load("cat_col_order.pkl")
num_col_order = joblib.load("num_col_order.pkl")
skewed_col_order = joblib.load("skewed_col_order.pkl")
threshold = joblib.load("threshold.pkl")
rare_maps = joblib.load("rare_maps.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cat_dims = [len(cat_maps[c]) for c in cat_col_order]
emb_dims = [min(50, (len(cat_maps[c]) + 1) // 2) for c in cat_col_order]

model = NN(num_numeric=len(num_col_order), cat_dims=cat_dims, emb_dims=emb_dims)
model.load_state_dict(torch.load("cr_weights.pth", map_location=device))
model = model.to(device)
model.eval()

app = FastAPI(title="Credit Risk Prediction API")

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

class InputData(BaseModel):
    data: Dict[str, Union[str, float, int, None]]

def preprocess_input(df: pd.DataFrame):

    df_num_raw = df[num_col_order].copy()
    df_num_imputed = pd.DataFrame(num_imputer.transform(df_num_raw), columns=num_col_order)

    df_num_scaled = pd.DataFrame(index=df.index)
    if skewed_col_order:
        df_num_scaled[skewed_col_order] = robust_scaler.transform(df_num_imputed[skewed_col_order])
    normal_col_order = [c for c in num_col_order if c not in skewed_col_order]
    if normal_col_order:
        df_num_scaled[normal_col_order] = std_scaler.transform(df_num_imputed[normal_col_order])

    x_num_tensor = torch.tensor(df_num_scaled.values, dtype=torch.float32).to(device)

    df_cat = df[cat_col_order].copy().astype(str)

    for col, rare_cats in rare_maps.items():
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].apply(lambda x: x if x not in rare_cats else 'Other')

    df_cat = df_cat.fillna("Other")

    for col in cat_col_order:
        df_cat[col] = df_cat[col].map(cat_maps[col]).fillna(0).astype(int)

    x_cat_tensor = torch.tensor(df_cat.values, dtype=torch.int64).to(device)

    return x_num_tensor, x_cat_tensor

def predict(df: pd.DataFrame, threshold=threshold):
    df = engineer_features(df, expected_num_cols=num_col_order, expected_cat_cols=cat_col_order)
    x_num, x_cat = preprocess_input(df)
    with torch.no_grad():
        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)
    return probs, preds

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    df = pd.DataFrame([input_data.data])
    probs, preds = predict(df)
    return {"probabilities": probs.tolist(), "predictions": preds.tolist()}