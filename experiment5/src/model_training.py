import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


def train_models(feature_path="results/final_pr_features.csv", metrics_path="results/model_metrics.csv", feature_importance_path="results/rf_feature_importances.csv", output_prefix=""):
    output_dir = os.path.dirname(metrics_path) or "results"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(feature_path, encoding="utf-8-sig")

    df = df.dropna(subset=["merged"]).copy()
    df["merged"] = df["merged"].astype(int)
    df["has_ai_reviewer"] = df["has_ai_reviewer"].astype(int)

    feature_cols = [
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
    ]

    X = df[feature_cols].fillna(0)
    y = df["merged"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

    print("开始训练 SVM 和 Random Forest...")
    svm.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)

    results = []
    for name, model, X_eval, _ in [
        ("SVM", svm, X_test_scaled, True),
        ("RandomForest", rf, X_test, False),
    ]:
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        results.append(metrics)

        report = classification_report(y_test, y_pred, zero_division=0)
        report_path = os.path.join(output_dir, f"{output_prefix}{name.lower()}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    feature_importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importances.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")

    print("模型训练与评估完成。")
    print(metrics_df.to_string(index=False))
    print(f"随机森林特征重要性已保存至 {feature_importance_path}")
    return metrics_df


if __name__ == "__main__":
    train_models()
