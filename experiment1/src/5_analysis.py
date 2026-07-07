import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# 中文显示配置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows用这个
# plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # Mac用这个
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
os.makedirs("images", exist_ok=True)

def safe_eval_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

if __name__ == "__main__":
    df = pd.read_csv("data/pr_feature_table.csv", encoding="utf-8-sig")
    print("===== 基础统计概览 =====")
    print(f"总PR数量：{len(df)}")
    print(f"平均PR长度：{df['pr_length'].mean():.0f} 字符")
    print(f"平均修改文件数：{df['file_count'].mean():.1f} 个")
    print(f"平均评审人数：{df['reviewer_count'].mean():.1f} 人")
    print(f"平均评论数：{df['comment_count'].mean():.1f} 条")
    print(f"整体合并率：{df['merged'].mean()*100:.1f}%")
    print(f"含AI评审的PR占比：{df['has_ai_reviewer'].mean()*100:.1f}%")
    print(f"含AI生成代码的PR占比：{df['has_ai_generated_code'].mean()*100:.1f}%")
    
    # ========== 1. Merge与Non-Merge数量统计（饼图） ==========
    merge_counts = df["merged"].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(merge_counts.values, 
            labels=["已合并", "未合并/关闭"], 
            autopct="%1.1f%%", 
            colors=["#4A90E2", "#E0E0E0"],
            startangle=90,
            textprops={"fontsize": 12})
    plt.title("PR合并状态分布", fontsize=14)
    plt.savefig("images/1_merge_status_pie.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 2. Review Comment数量分布（直方图） ==========
    plt.figure(figsize=(10, 6))
    # 截取95分位以内，避免极端值影响展示
    threshold = df["comment_count"].quantile(0.95)
    sns.histplot(df[df["comment_count"] <= threshold]["comment_count"], 
                 bins=30, kde=True, color="#4A90E2")
    plt.xlabel("评审评论数量", fontsize=12)
    plt.ylabel("PR数量", fontsize=12)
    plt.title("PR评审评论数量分布", fontsize=14)
    plt.savefig("images/2_comment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 3. Label分布（柱状图Top10） ==========
    all_labels = []
    for labels_str in df["labels"]:
        all_labels.extend(safe_eval_list(labels_str))
    
    if all_labels:
        label_top10 = pd.Series(all_labels).value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=label_top10.values, y=label_top10.index, palette="Blues_r")
        plt.xlabel("出现次数", fontsize=12)
        plt.ylabel("标签名称", fontsize=12)
        plt.title("PR标签出现频次Top10", fontsize=14)
        plt.savefig("images/3_label_top10.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # ========== 4. Reviewer数量分布（直方图） ==========
    plt.figure(figsize=(10, 6))
    threshold = df["reviewer_count"].quantile(0.95)
    sns.histplot(df[df["reviewer_count"] <= threshold]["reviewer_count"], 
                 bins=15, kde=True, color="#5BA3E0")
    plt.xlabel("评审者数量", fontsize=12)
    plt.ylabel("PR数量", fontsize=12)
    plt.title("PR评审者数量分布", fontsize=14)
    plt.savefig("images/4_reviewer_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 5. PR长度分布（直方图） ==========
    plt.figure(figsize=(10, 6))
    threshold = df["pr_length"].quantile(0.95)
    sns.histplot(df[df["pr_length"] <= threshold]["pr_length"], 
                 bins=50, kde=True, color="#7BB7ED")
    plt.xlabel("PR文本长度（字符数）", fontsize=12)
    plt.ylabel("PR数量", fontsize=12)
    plt.title("PR文本长度分布", fontsize=14)
    plt.savefig("images/5_pr_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 6. AI与Human PR数量对比（柱状图） ==========
    ai_counts = df["has_ai_generated_code"].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["人工提交PR", "含AI生成代码PR"], 
                y=[ai_counts.get(False, 0), ai_counts.get(True, 0)],
                palette=["#E0E0E0", "#4A90E2"])
    plt.ylabel("PR数量", fontsize=12)
    plt.title("AI生成PR与人工PR数量对比", fontsize=14)
    for i, v in enumerate([ai_counts.get(False, 0), ai_counts.get(True, 0)]):
        plt.text(i, v + 10, str(v), ha="center", fontsize=11)
    plt.savefig("images/6_ai_vs_human.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\n所有图表已生成，保存在 images/ 文件夹下")