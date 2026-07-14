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


def clean_data():
    os.makedirs("results", exist_ok=True)

    pr_basic = pd.read_csv("results/all_pr_basic.csv", encoding="utf-8-sig")
    pr_code = pd.read_csv("results/all_code_changes.csv", encoding="utf-8-sig")
    pr_feature = pd.read_csv("results/pr_feature_table.csv", encoding="utf-8-sig")

    pr_feature["merged"] = pr_feature["merged"].astype(bool)
    pr_feature["has_ai_generated_code"] = pr_feature["has_ai_generated_code"].astype(bool)

    # 保留人工编写代码的 PR
    clean_df = pr_feature[pr_feature["has_ai_generated_code"] == False].copy()
    clean_df = clean_df.drop_duplicates(subset=["repo", "pr_id"])

    clean_df["labels"] = clean_df["labels"].fillna("[]")
    clean_df["modified_files"] = clean_df["modified_files"].fillna("[]")
    clean_df["modified_functions"] = clean_df["modified_functions"].fillna("[]")
    clean_df["commit_messages"] = clean_df["commit_messages"].fillna("[]")

    clean_df.to_csv("results/clean_pr_data.csv", index=False, encoding="utf-8-sig")

    summary = {
        "total_pr": len(pr_feature),
        "human_pr": len(clean_df),
        "merged_ratio": clean_df["merged"].mean()
    }

    print("数据清洗完成。")
    print(f"总 PR: {summary['total_pr']}，人工 PR: {summary['human_pr']}，人工 PR 合并率: {summary['merged_ratio']:.3f}")
    return clean_df


if __name__ == "__main__":
    clean_data()
