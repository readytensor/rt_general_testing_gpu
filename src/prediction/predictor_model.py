import os
import warnings
import torch
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from config import paths
from pathlib import Path
from logger import get_logger
from utils import list_image_files


warnings.filterwarnings("ignore")

logger = get_logger(task_name="model")

gpu_available = torch.cuda.is_available()

logger.info(f"GPU available: {gpu_available}")


class PredictorModel:
    """
    This class provides a consistent interface that can be used with other models.
    """

    MODEL_NAME = "predictor"

    def __init__(
        self,
        data_schema: Dict,
        **kwargs,
    ):
        self.data_schema = data_schema
        if data_schema is not None:
            self.model_category = data_schema["modelCategory"]
        else:
            self.model_category = "image_classification"

        self.kwargs = kwargs

    def fit_regression(self, train_data: pd.DataFrame) -> None:
        self.mean = train_data[self.data_schema["target"]["name"]].mean()

    def regression_predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        test_data["prediction"] = self.mean
        noise = np.random.normal(0, 1, size=len(test_data))
        test_data["prediction"] += noise
        return test_data

    def fit_classification(self, train_data: pd.DataFrame) -> None:
        return None

    def classification_predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        id_col = self.data_schema["id"]["name"]
        classes = self.data_schema["target"]["classes"]
        n_samples = test_data.shape[0]
        # Generate a random m x n array
        rand_array = np.random.rand(n_samples, len(classes))

        # Normalize each row to make their sum equal 1
        normalized_array = rand_array / rand_array.sum(axis=1, keepdims=True)
        test_data[classes] = normalized_array
        return test_data[[id_col] + classes]

    def fit_forecasting(self, train_data: pd.DataFrame) -> None:
        id_col = self.data_schema["idField"]["name"]
        target_col = self.data_schema["forecastTarget"]["name"]
        grouped = train_data.groupby(id_col)

        all_ids = [i for i, _ in grouped]
        all_series = [i for _, i in grouped]
        means = [i[target_col].mean() for i in all_series]
        self.mean = {i: j for i, j in zip(all_ids, means)}

    def forecasting_predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        id_col = self.data_schema["idField"]["name"]
        test_data["prediction"] = test_data[id_col].map(self.mean)
        return test_data

    def fit_img_classification(self) -> None:
        classes = [
            i
            for i in os.listdir(paths.TRAIN_DIR_IMG_CLASS)
            if os.path.isdir(os.path.join(paths.TRAIN_DIR_IMG_CLASS, i))
        ]
        self.classes = classes

    def img_classification_predict(self) -> pd.DataFrame:
        ids = [Path(i).name for i in list_image_files(paths.TEST_DIR_IMG_CLASS)]
        n_samples = len(ids)
        # Generate a random m x n array
        rand_array = np.random.rand(n_samples, len(self.classes))
        # Normalize each row to make their sum equal 1
        normalized_array = rand_array / rand_array.sum(axis=1, keepdims=True)

        predictions_df = pd.DataFrame(normalized_array, columns=self.classes)
        predictions_df["id"] = ids
        return predictions_df

    def save(self, predictor_dir_path: str) -> None:
        model_path = os.path.join(predictor_dir_path, "predictor.joblib")
        joblib.dump(self, model_path)

    @classmethod
    def load(cls, predictor_dir_path: str) -> "PredictorModel":
        model_path = os.path.join(predictor_dir_path, "predictor.joblib")
        return joblib.load(model_path)

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(data_schema: str, train_data: pd.DataFrame) -> PredictorModel:
    model = PredictorModel(data_schema=data_schema)
    if model.model_category == "regression":
        model.fit_regression(train_data=train_data)
    elif model.model_category == "classification":
        model.fit_classification(train_data=train_data)
    elif model.model_category == "forecasting":
        model.fit_forecasting(train_data=train_data)
    elif model.model_category == "image_classification":
        model.fit_img_classification()
    return model


def predict_with_model(model: PredictorModel, test_data: pd.DataFrame) -> pd.DataFrame:
    if model.model_category == "regression":
        pred = model.regression_predict(test_data=test_data)
    elif model.model_category in ["binary_classification", "multiclass_classification"]:
        pred = model.classification_predict(test_data=test_data)
    elif model.model_category == "forecasting":
        pred = model.forecasting_predict(test_data=test_data)
    elif model.model_category == "image_classification":
        pred = model.img_classification_predict()

    return pred


def save_predictor_model(model: PredictorModel, predictor_dir_path: str) -> None:
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> PredictorModel:
    return PredictorModel.load(predictor_dir_path)
