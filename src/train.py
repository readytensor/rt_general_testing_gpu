import os
from config import paths
from logger import get_logger, log_error
from prediction.predictor_model import save_predictor_model, train_predictor_model
from utils import (
    read_json_as_dict,
    set_seeds,
    read_csv_in_directory,
    ResourceTracker,
)

logger = get_logger(task_name="train")

SCHEMA_EXISTS = os.path.exists(paths.SCHEMA_DIR)


def run_training(
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir_path: str = paths.TRAIN_DIR,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    data_schema_dir_path: str = paths.SCHEMA_DIR,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        model_config_file_path (str, optional): The path of the model configuration file.
        train_dir_path (str, optional): The directory path of the train data.
        predictor_dir_path (str, optional): The directory path where to save the predictor model.
        data_schema_dir_path (str, optional): The directory path of the data schema.
    Returns:
        None
    """

    try:
        with ResourceTracker(logger=logger):
            logger.info("Starting training...")

            # load model config
            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)
            if SCHEMA_EXISTS:
                logger.info("Loading schema...")
                data_schema = read_json_as_dict(data_schema_dir_path)
                logger.info("Loading input training...")
                train_data = read_csv_in_directory(train_dir_path)
            else:
                data_schema = None
                train_data = None

            # set seeds
            logger.info("Setting seeds...")
            set_seeds(seed_value=model_config["seed_value"])

            logger.info(f"Training model...")
            model = train_predictor_model(
                data_schema=data_schema,
                train_data=train_data,
            )

        # save predictor model
        logger.info("Saving model...")
        save_predictor_model(model, predictor_dir_path)

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
