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


def clean_data(ai_generated_only=False, output_path=None):
    os.makedirs("results", exist_ok=True)

    if output_path is None:
        output_path = "results/ai_clean_pr_data.csv" if ai_generated_only else "results/clean_pr_data.csv"

    pr_feature = pd.read_csv("results/pr_feature_table.csv", encoding="utf-8-sig")

    pr_feature["merged"] = pr_feature["merged"].astype(bool)
    pr_feature["has_ai_generated_code"] = pr_feature["has_ai_generated_code"].astype(bool)

    subset_name = "AI生成代码" if ai_generated_only else "人工编写代码"
    clean_df = pr_feature[pr_feature["has_ai_generated_code"] == ai_generated_only].copy()
    clean_df = clean_df.drop_duplicates(subset=["repo", "pr_id"])

    clean_df["labels"] = clean_df["labels"].fillna("[]")
    clean_df["modified_files"] = clean_df["modified_files"].fillna("[]")
    clean_df["modified_functions"] = clean_df["modified_functions"].fillna("[]")
    clean_df["commit_messages"] = clean_df["commit_messages"].fillna("[]")

    clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "total_pr": len(pr_feature),
        f"{subset_name.lower()}_pr": len(clean_df),
        "subset_ratio": len(clean_df) / len(pr_feature) if len(pr_feature) else 0.0,
        "merged_ratio": clean_df["merged"].mean(),
    }

    print("数据清洗完成。")
    print(f"总 PR: {summary['total_pr']}，{subset_name} PR: {summary[f'{subset_name.lower()}_pr']}，合并率: {summary['merged_ratio']:.3f}")
    return clean_df


if __name__ == "__main__":
    clean_data()
