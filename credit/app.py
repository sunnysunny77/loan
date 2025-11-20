import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pydantic import BaseModel
from typing import Dict, Union

app = FastAPI(title="Credit Risk Prediction API")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


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
    
    TotalPastDueLog = np.log1p(TotalPastDue)
    
    RevolvingUtilizationCapped = df_e["RevolvingUtilizationOfUnsecuredLines"].clip(upper=5.0)
    RevolvingUtilizationFilled = RevolvingUtilizationCapped.fillna(0)
    RevolvingUtilizationCappedLog = np.log1p(RevolvingUtilizationFilled)
    RevolvingUtilizationCappedLogSafe = RevolvingUtilizationCappedLog.replace(0, np.nan)
        
    AgeSafe = df_e["age"].replace(0, np.nan)

    DebtRatioCapped = df_e["DebtRatio"].fillna(0).clip(upper=10000.0)
        
    DebtRatioCappedLog = np.log1p(DebtRatioCapped)
    
    DebtRatioSafe =  DebtRatioCappedLog.replace(0, np.nan)

    MonthlyIncomeLog = np.log1p(df_e["MonthlyIncome"].fillna(0))

    MonthlyIncomeSafe = MonthlyIncomeLog.replace(0, np.nan)

    CreditLinesSafe = df_e["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan)

    NumberRealEstateLoansOrLinesfilled = df_e["NumberRealEstateLoansOrLines"].fillna(0)

    DebtToIncome = DebtRatioSafe * MonthlyIncomeSafe

    DelinquencyScore = (
        NumberOfTime3059DaysPastDueNotWorse +
        NumberOfTime6089DaysPastDueNotWorse * 2 +
        NumberOfTimes90DaysLate * 3
    )

    PastDueSeverity = (
        NumberOfTime3059DaysPastDueNotWorse*1 +
        NumberOfTime6089DaysPastDueNotWorse*3 +
        NumberOfTimes90DaysLate*6
    )

    PastDueSeverityLog = np.log1p(PastDueSeverity)

    DelinquencyScoreLog = np.log1p(DelinquencyScore)

    CreditBurdenIndex = (
        RevolvingUtilizationCappedLogSafe +
        DebtRatioSafe +
        2 * DelinquencyScoreLog
    )

    RepaymentCapacity = MonthlyIncomeSafe - (
        DebtRatioSafe * MonthlyIncomeSafe + 0.5 * PastDueSeverityLog
    )
    

    df_e["DebtRatio"] = DebtRatioSafe
    df_e["MonthlyIncome"] = MonthlyIncomeSafe 
    df_e["RevolvingUtilization"] = RevolvingUtilizationCappedLogSafe
    df_e["AgeSafe"] = AgeSafe
    df_e["CreditBurdenIndex"] = CreditBurdenIndex
    df_e["RepaymentCapacity"] = RepaymentCapacity

    df_e["DelinquencyScore"] = DelinquencyScoreLog
    df_e["PastDueSeverity"] = PastDueSeverityLog
    df_e["AgeDelinquencyInteraction"] = AgeSafe * DelinquencyScoreLog
    df_e["LatePaymentsPerCreditLine"] = TotalPastDueLog / CreditLinesSafe 
    df_e["DebtToIncome"] = DebtToIncome
    df_e["ExposureToDelinquency"] = DebtRatioSafe / (1 + DelinquencyScoreLog)
    df_e["RealEstateLeveragePerAge"] =  (NumberRealEstateLoansOrLinesfilled * RevolvingUtilizationCappedLogSafe) / AgeSafe
    df_e["CreditLinesSafeAge"] =  CreditLinesSafe / AgeSafe
    df_e["IncomePerAge"] = MonthlyIncomeSafe / AgeSafe

    Utilization_bins = [-0.01, 0.1, 0.3, 0.6, 0.9, 1.5, 10]
    Utilization_labels = ["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"]
    UtilizationBucket = pd.cut(RevolvingUtilizationFilled, bins=Utilization_bins, labels=Utilization_labels)

    Late_bins = [-1, 0, 1, 3, 6, np.inf]
    Late_labels = ["NoLate", "FewLate", "ModerateLate", "FrequentLate", "ChronicLate"]
    LatePaymentBucket = pd.cut(TotalPastDueLog, bins=Late_bins, labels=Late_labels)

    df_e["UtilizationBucketLateBucket"] = (
        UtilizationBucket.astype(str) + "_" + LatePaymentBucket.astype(str)
    )

    engineered_cols = [
        "DebtRatio",
        "RevolvingUtilization",
        "MonthlyIncome",
        "DelinquencyScore",
        "LatePaymentsPerCreditLine",
        "UtilizationBucketLateBucket",
        "CreditLinesSafeAge",
        "PastDueSeverity",
        "RealEstateLeveragePerAge",
        "AgeSafe",
        "IncomePerAge",
        "DebtToIncome",
        "CreditBurdenIndex",
        "AgeDelinquencyInteraction",
        "RepaymentCapacity",
        "ExposureToDelinquency",
    ]

    engineered_df = df_e[engineered_cols]

    return engineered_df

class NN(nn.Module):
    def __init__(self, input_dim): 
        super().__init__()
        self.bn_all = nn.BatchNorm1d(input_dim)
        
        self.input_dim = input_dim 

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

        self.out = nn.Linear(64, 1)

    def forward(self, x_all): 
    
        x = self.bn_all(x_all) 

        x_main = self.main(x)

        x_skip = self.skip_proj_main(x)

        x_combined = x_main + x_skip
        
        return self.out(x_combined).squeeze(1)

model_b = xgb.XGBClassifier()
model_b.load_model("cr_b.json")
xgb_col_order = joblib.load("xgb_col_order.pkl")
nn_col_order = joblib.load("nn_col_order.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
robust_scaler = joblib.load("robust_scaler.pkl")
std_scaler = joblib.load("std_scaler.pkl")
cat_maps = joblib.load("cat_maps.pkl")
cat_col_order = joblib.load("cat_col_order.pkl")
num_col_order = joblib.load("num_col_order.pkl")
skewed_col_order = joblib.load("skewed_col_order.pkl")
threshold_a = joblib.load("threshold_a.pkl")
threshold_b = joblib.load("threshold_b.pkl")
rare_maps = joblib.load("rare_maps.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN(input_dim=len(nn_col_order))
weights_path = "cr_weights.pth"
loaded_weights = torch.load(weights_path, map_location=device) 
model.load_state_dict(loaded_weights)
model = model.to(device)
model.eval()

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
            df_copy[col] = (
                df_copy[col].astype(str)
                .map(cat_maps.get(col, {}))
                .fillna(0)
                .astype(int)
            )
        df_final = df_copy.reindex(columns=xgb_col_order, fill_value=0.0).astype(np.float32)
        return df_final
    else: 
        df_copy = df_copy.reindex(columns=nn_col_order, fill_value=0.0)
        x_tensor = torch.tensor(df_copy.astype("float32").values, dtype=torch.float32).to(device)
        return x_tensor

def predict_nn(df: pd.DataFrame, threshold=threshold_a):
    x_all = preprocess(df, for_xgb=False)
    with torch.no_grad():
        logits = model(x_all)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)
    return probs, preds

@app.post("/predict_nn")
def predict_endpoint(input_data: InputData):
    df = pd.DataFrame([input_data.data]).replace("", np.nan)
    if df.isna().all(axis=1).iloc[0]:
        return {"message": "No input provided"}
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
    if df.isna().all(axis=1).iloc[0]:
        return {"message": "No input provided"}
    probs, preds = predict_xgb(df)
    return {"probabilities": probs.tolist(), "predictions": preds.tolist()}