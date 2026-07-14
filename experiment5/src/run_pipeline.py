from syc.data_cleaning import clean_data
from syc.ir_feature_generation import generate_ir_features
from syc.feature_extraction import extract_features
from syc.model_training import train_models


def run_pipeline(ai_generated_only=False):
    clean_path = "results/ai_clean_pr_data.csv" if ai_generated_only else "results/clean_pr_data.csv"
    feature_path = "results/ai_final_pr_features.csv" if ai_generated_only else "results/final_pr_features.csv"
    metrics_path = "results/ai_model_metrics.csv" if ai_generated_only else "results/model_metrics.csv"
    importance_path = "results/ai_rf_feature_importances.csv" if ai_generated_only else "results/rf_feature_importances.csv"
    prefix = "ai_" if ai_generated_only else ""

    clean_data(ai_generated_only=ai_generated_only, output_path=clean_path)
    ir_path = "results/ai_pr_ast_cfg_features.csv" if ai_generated_only else "results/pr_ast_cfg_features.csv"
    generate_ir_features(input_path=clean_path, output_path=ir_path)
    extract_features(input_path=clean_path, ir_path=ir_path, output_path=feature_path)
    train_models(feature_path=feature_path, metrics_path=metrics_path, feature_importance_path=importance_path, output_prefix=prefix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PR merge-prediction pipeline for human or AI-generated-code subsets.")
    parser.add_argument("--ai-generated-only", action="store_true", help="Filter to PRs where AI-generated code is present.")
    args = parser.parse_args()
    run_pipeline(ai_generated_only=args.ai_generated_only)
