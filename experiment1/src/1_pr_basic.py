import requests
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time

# 加载Token
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# 5个目标开源项目
REPOS = [
    "microsoft/vscode",
    "pytorch/pytorch",
    "langchain-ai/langchain",
    "facebook/react",
    "apache/spark"
]

def get_pr_basic(repo_full_name, pr_count=300):
    owner, repo = repo_full_name.split("/")
    pr_list = []
    pages = pr_count // 100  # 每页100条，共3页
    
    for page in tqdm(range(1, pages+1), desc=f"采集 {repo} PR基础数据"):
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {
            "state": "all",
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "desc"
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        
        # 处理请求频率限制
        if response.status_code == 429:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 60))
            wait = reset_time - int(time.time()) + 2
            print(f"触发限流，等待 {wait} 秒")
            time.sleep(wait)
            response = requests.get(url, headers=HEADERS, params=params)
        
        if response.status_code != 200:
            print(f"请求失败，状态码：{response.status_code}")
            continue
        
        prs = response.json()
        for pr in prs:
            pr_list.append({
                "repo": repo_full_name,
                "pr_id": pr["number"],
                "title": pr["title"],
                "body": pr["body"] if pr["body"] else "",
                "author": pr["user"]["login"],
                "created_time": pr["created_at"],
                "merged": pr.get("merged_at") is not None,
                "labels": [label["name"] for label in pr["labels"]]
            })
        time.sleep(0.5)  # 礼貌延迟，避免触发风控
    
    return pd.DataFrame(pr_list)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    all_data = []
    
    for repo in REPOS:
        df = get_pr_basic(repo)
        all_data.append(df)
        # 每个项目单独保存一份
        df.to_csv(f"data/{repo.replace('/', '_')}_pr_basic.csv", index=False, encoding="utf-8-sig")
    
    # 合并保存总表
    total_df = pd.concat(all_data, ignore_index=True)
    total_df.to_csv("data/all_pr_basic.csv", index=False, encoding="utf-8-sig")
    print(f"PR基础数据采集完成，共 {len(total_df)} 条，已保存到 data/all_pr_basic.csv")