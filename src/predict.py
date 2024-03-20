import os
from config import paths
from logger import get_logger, log_error
from prediction.predictor_model import predict_with_model, load_predictor_model
from utils import (
    save_dataframe_as_csv,
    read_csv_in_directory,
    ResourceTracker,
)

logger = get_logger(task_name="predict")

SCHEMA_EXISTS = os.path.exists(paths.SCHEMA_DIR)


def run_batch_predictions(
    test_dir_path: str = paths.TEST_DIR,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    Args:
        test_dir_path (str): Directory path for the test data.
        predictor_dir_path (str): Path to the directory of saved model.
        predictions_file_path (str): Path where the predictions file will be saved.
    """

    try:
        with ResourceTracker(logger):
            logger.info("Making batch predictions...")

            test_data = None
            if SCHEMA_EXISTS:
                logger.info("Loading test data...")
                test_data = read_csv_in_directory(test_dir_path)

            logger.info("Loading predictor model...")
            predictor_model = load_predictor_model(predictor_dir_path)

            logger.info("Making predictions...")
            predictions_df = predict_with_model(predictor_model, test_data)

        logger.info("Saving predictions dataframe...")
        save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()
