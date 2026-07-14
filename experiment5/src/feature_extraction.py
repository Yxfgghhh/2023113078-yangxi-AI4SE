import ast
import os
import pandas as pd


def safe_eval_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def extract_features(input_path="results/clean_pr_data.csv", ir_path="results/pr_ast_cfg_features.csv", output_path="results/final_pr_features.csv"):
    os.makedirs("results", exist_ok=True)
    clean_df = pd.read_csv(input_path, encoding="utf-8-sig")
    ir_df = pd.read_csv(ir_path, encoding="utf-8-sig")

    feature_df = clean_df.merge(ir_df, on=["repo", "pr_id"], how="left")

    feature_df["commit_messages"] = feature_df["commit_messages"].apply(safe_eval_list)
    feature_df["modified_functions"] = feature_df["modified_functions"].apply(safe_eval_list)
    feature_df["label_count"] = feature_df["label_count"].fillna(0).astype(int)
    feature_df["reviewer_count"] = feature_df["reviewer_count"].fillna(0).astype(int)
    feature_df["comment_count"] = feature_df["comment_count"].fillna(0).astype(int)
    feature_df["has_ai_reviewer"] = feature_df["has_ai_reviewer"].fillna(False).astype(bool)

    feature_df["commit_message_length"] = feature_df["commit_messages"].apply(lambda msgs: sum(len(str(m)) for m in msgs))
    feature_df["pr_description_length"] = feature_df["body"].fillna("").astype(str).apply(len)
    feature_df["modified_function_count"] = feature_df["modified_functions"].apply(len)
    feature_df["pr_length"] = feature_df["pr_length"].fillna(0).astype(int)

    output_columns = [
        "repo",
        "pr_id",
        "file_count",
        "total_additions",
        "total_deletions",
        "commit_message_length",
        "pr_description_length",
        "reviewer_count",
        "comment_count",
        "has_ai_reviewer",
        "label_count",
        "pr_length",
        "modified_function_count",
        "ast_node_count",
        "cfg_node_count",
        "ast_density",
        "cfg_density",
        "merged",
    ]

    final_df = feature_df[output_columns].copy()
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"特征提取完成，已保存 {output_path}")
    print(f"特征数量：{len(final_df)}, 特征列：{output_columns}")
    return final_df


if __name__ == "__main__":
    extract_features()
