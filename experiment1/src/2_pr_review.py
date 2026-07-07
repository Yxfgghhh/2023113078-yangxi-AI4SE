import requests
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# AI评审者关键词（用户名包含即判定为AI/机器人）
AI_KEYWORDS = {"bot", "ai", "copilot", "coderabbit", "codacy", 
               "codecov", "github-actions", "reviewdog", "sonarcloud", "dependabot"}

def is_ai_reviewer(reviewer_name):
    name_lower = reviewer_name.lower()
    return any(keyword in name_lower for keyword in AI_KEYWORDS)

def get_pr_reviews(repo_full_name, pr_id):
    owner, repo = repo_full_name.split("/")
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_id}/reviews"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []
        
        reviews = response.json()
        result = []
        for review in reviews:
            if not review.get("user"):
                continue
            result.append({
                "repo": repo_full_name,
                "pr_id": pr_id,
                "reviewer": review["user"]["login"],
                "review_decision": review["state"],
                "review_time": review["submitted_at"],
                "review_comment": review["body"] if review["body"] else ""
            })
        return result
    except Exception as e:
        print(f"PR {pr_id} 采集失败：{e}")
        return []

if __name__ == "__main__":
    # 读取上一步的PR基础数据
    pr_df = pd.read_csv("data/all_pr_basic.csv", encoding="utf-8-sig")
    all_reviews = []
    
    for _, row in tqdm(pr_df.iterrows(), total=len(pr_df), desc="采集评审数据"):
        reviews = get_pr_reviews(row["repo"], row["pr_id"])
        all_reviews.extend(reviews)
        time.sleep(0.2)
    
    review_df = pd.DataFrame(all_reviews)
    if len(review_df) > 0:
        review_df["is_ai_reviewer"] = review_df["reviewer"].apply(is_ai_reviewer)
    
    review_df.to_csv("data/all_pr_reviews.csv", index=False, encoding="utf-8-sig")
    print(f"评审数据采集完成，共 {len(review_df)} 条评审记录，已保存到 data/all_pr_reviews.csv")