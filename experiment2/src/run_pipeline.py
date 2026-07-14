from syc.data_cleaning import clean_data
from syc.ir_feature_generation import generate_ir_features
from syc.feature_extraction import extract_features
from syc.model_training import train_models


if __name__ == "__main__":
    clean_data()
    generate_ir_features()
    extract_features()
    train_models()
