from xgboost import XGBRegressor
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd
from lightautoml.tasks import Task
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
class SIModel(XGBRegressor, TabularAutoML):
    def __init__(self):
        self.cc50_reg = XGBRegressor()
        self.ic50_reg = TabularAutoML(
                            task = Task(
                                name = 'reg',
                                metric = r2_score),
                            timeout=1000
                        )
        self.cc50_reg = joblib.load("reg_cc50_xgboost_1.joblib")
        self.ic50_reg = joblib.load("automl_ic50.joblib")
        self.encoder_ic50 = OneHotEncoder()
        self.encoder_cc50 = OneHotEncoder()
        self.encoder_cc50 = joblib.load("onehot_encoder_cc50 (1).joblib")
        self.encoder_ic50 = joblib.load("onehot_encoder_ic50.joblib")
        self.ic50_scaler = MinMaxScaler()
        self.cc50_scaler = MinMaxScaler()
        self.ic50_target_scaler = StandardScaler()
        self.cc50_target_scaler = StandardScaler()
        self.ic50_scaler = joblib.load("scaler_ic50.joblib")
        self.cc50_scaler = joblib.load("scaler_cc50.joblib")
        self.ic50_target_scaler = joblib.load("target_scaler_ic50.joblib")
        self.cc50_target_scaler = joblib.load("target_scaler_cc50.joblib")
    def predict(self, data):
        
        cat_cols = ["Strain", "Cell"]
        data_cc50 = data.copy()
        data_ic50 = data.copy()
        data_ic50[self.ic50_scaler.feature_names_in_] = pd.DataFrame(self.ic50_scaler.transform(data[self.ic50_scaler.feature_names_in_]), columns=self.ic50_scaler.feature_names_in_)
        data_cc50[self.cc50_scaler.feature_names_in_] = pd.DataFrame(self.cc50_scaler.transform(data[self.cc50_scaler.feature_names_in_]), columns=self.cc50_scaler.feature_names_in_)
        for col in cat_cols:
            data_ic50[col] = data_ic50[col].apply(lambda x: "unknown" if col + "_" + x not in self.encoder_ic50.get_feature_names_out() else x)
            data_cc50[col] = data_cc50[col].apply(lambda x: "unknown" if col + "_" + x not in self.encoder_cc50.get_feature_names_out() else x)
        arr_cc50 = pd.DataFrame(self.encoder_cc50.transform(pd.DataFrame(data_cc50[cat_cols])).toarray(), columns=self.encoder_cc50.get_feature_names_out())
        data_cc50 = arr_cc50.join(data_ic50)
        data_cc50.drop(columns=cat_cols, inplace=True)
        
        arr_ic50 = pd.DataFrame(self.encoder_ic50.transform(pd.DataFrame(data_ic50[cat_cols])).toarray(), columns=self.encoder_ic50.get_feature_names_out())
        data_ic50 = arr_ic50.join(data_ic50)
        data_ic50.drop(columns=cat_cols, inplace=True)
        
        ic50 = self.ic50_target_scaler.inverse_transform(np.abs(self.ic50_reg.predict(data_ic50).data.reshape((-1, 1))))
        cc50 = self.cc50_target_scaler.inverse_transform(np.abs(self.cc50_reg.predict(data_cc50).reshape((-1, 1))))
        si = cc50 / ic50
        preds = pd.DataFrame(np.concatenate((cc50, ic50, si), axis=1), columns=["CC50, mg/ml", "IC50, mg/ml", "SI"])
        return preds
    