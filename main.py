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

    NumberOfTime3059DaysPastDueNotWorse = df_e["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
    NumberOfTimes90DaysLate = df_e["NumberOfTimes90DaysLate"].fillna(0)
    NumberOfTime6089DaysPastDueNotWorse = df_e["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)

    TotalPastDue = (
        NumberOfTime3059DaysPastDueNotWorse
        + NumberOfTimes90DaysLate
        + NumberOfTime6089DaysPastDueNotWorse
    )

    df_e["MajorDelinquencyBinary"] = (
        (NumberOfTimes90DaysLate > 0) |
        (NumberOfTime6089DaysPastDueNotWorse > 0)
    ).astype(int)
    
    df_e["TotalPastDue"] = TotalPastDue

    RevolvingUtilizationOfUnsecuredLines = np.log1p(df_e["RevolvingUtilizationOfUnsecuredLines"].fillna(0))

    HasDelinquencies = (TotalPastDue > 0).astype(int)
    
    df_e["TotalPastDue_Squared"] = TotalPastDue ** 2

    df_e['90DaysLate_Squared'] = NumberOfTimes90DaysLate ** 2
    
    df_e["HasDelinquencies"] = HasDelinquencies

    df_e["NormalizedUtilization"] = np.sqrt(RevolvingUtilizationOfUnsecuredLines)
    
    df_e["DelinquencyInteraction"] = TotalPastDue * RevolvingUtilizationOfUnsecuredLines
    
    age_safe = df_e["age"].fillna(0)
    
    MonthlyIncome_safe = np.log1p(df_e["MonthlyIncome"].fillna(1.0))
    
    DebtRatio_safe = np.log1p(df_e["DebtRatio"].fillna(0))

    CreditLinesSafe = df_e["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan)

    df_e["UtilizationPerAge"] = RevolvingUtilizationOfUnsecuredLines / (age_safe + 1)
    
    df_e["LatePaymentsPerAge"] = TotalPastDue / (age_safe + 1)
    
    df_e["LatePaymentsPerCreditLine"] = TotalPastDue / CreditLinesSafe
    
    IncomePerCreditLine = MonthlyIncome_safe / CreditLinesSafe

    df_e["IncomePerCreditLine"] = IncomePerCreditLine
    
    df_e["DebtToIncome"] = DebtRatio_safe * MonthlyIncome_safe

    AgeRisk = np.where(df_e["age"].fillna(0) < 25, 1,
                     np.where(df_e["age"].fillna(0) < 35, 0.8,
                     np.where(df_e["age"].fillna(0) < 50, 0.6, 0.4)))

    df_e["AgeRisk"] = AgeRisk

    df_e["DebtToIncomeAgeRisk"] = df_e["DebtToIncome"] * AgeRisk     

    df["IncomePerCreditLine_HasDelinquencies"] = IncomePerCreditLine * HasDelinquencies
    
    utilization_bins = [-0.01, 0.1, 0.3, 0.6, 0.9, 1.5, 10]
    utilization_labels = ["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"]
    df_e["UtilizationBucket"] = pd.cut(RevolvingUtilizationOfUnsecuredLines, bins=utilization_bins, labels=utilization_labels)
    
    age_bins = [0, 25, 50, 120] 
    age_labels = ["Young", "Mid", "Senior"]
    df_e["AgeBucket"] = pd.cut(age_safe, bins=age_bins, labels=age_labels)

    late_bins = [-1, 0, 1, 3, 6, np.inf]
    late_labels = ["NoLate", "FewLate", "ModerateLate", "FrequentLate", "ChronicLate"]
    df_e["LatePaymentBucket"] = pd.cut(TotalPastDue, bins=late_bins, labels=late_labels)

    df_e = df_e.drop(
        ["RevolvingUtilizationOfUnsecuredLines", 
         "NumberOfTimes90DaysLate",
         "NumberRealEstateLoansOrLines",
         "DebtRatio",
         "MonthlyIncome", 
         "NumberOfOpenCreditLinesAndLoans",
         "NumberOfTime30-59DaysPastDueNotWorse",
         "NumberOfTime60-89DaysPastDueNotWorse",
         "age",
         "NumberOfDependents",
        ], axis=1, errors='ignore')

    return df_e

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
num_col_order = joblib.load("num_col_order.pkl")
skewed_col_order = joblib.load("skewed_col_order.pkl")
threshold_a = joblib.load("threshold_a.pkl")
threshold_b = joblib.load("threshold_b.pkl")
rare_maps = joblib.load("rare_maps.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cat_dims = [len(cat_maps[c]) for c in cat_col_order]
emb_dims = [min(50, (len(cat_maps[c]) + 1) // 2) for c in cat_col_order]

model = NN(num_numeric=len(num_col_order), cat_dims=cat_dims, emb_dims=emb_dims)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        df_final = df_copy.astype(np.float32)
        trained_features = model_b.get_booster().feature_names
        df_final = df_final.reindex(columns=trained_features, fill_value=0.0)
        return df_final
    else:
        for col in cat_col_order:
            df_copy[col] = df_copy[col].astype(str).map(cat_maps[col]).astype(int)
        x_num_tensor = torch.tensor(df_copy[num_col_order].values, dtype=torch.float32).to(device)
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