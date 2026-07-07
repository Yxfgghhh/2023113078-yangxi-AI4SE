import requests
import os
import re
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

# 匹配函数声明的正则（适配Python/Java/JS/C++/Go等主流语言）
FUNCTION_PATTERNS = [
    re.compile(r'^\+\s*(?:def|function|func|fn)\s+(\w+)\s*\('),
    re.compile(r'^\+\s*(?:public|private|protected|static)?\s*[\w<>\[\]]+\s+(\w+)\s*\(.*\)'),
]

def extract_modified_functions(patch):
    if not patch:
        return []
    functions = set()
    for line in patch.split("\n"):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pattern in FUNCTION_PATTERNS:
            match = pattern.search(line)
            if match:
                functions.add(match.group(1))
    return list(functions)

def get_pr_files(repo_full_name, pr_id):
    owner, repo = repo_full_name.split("/")
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_id}/files"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return []
        files = response.json()
        result = []
        for f in files:
            funcs = extract_modified_functions(f.get("patch", ""))
            result.append({
                "filename": f["filename"],
                "patch": f.get("patch", ""),
                "modified_functions": funcs,
                "additions": f["additions"],
                "deletions": f["deletions"]
            })
        return result
    except Exception as e:
        print(f"PR {pr_id} 文件采集失败：{e}")
        return []

def get_pr_commits(repo_full_name, pr_id):
    owner, repo = repo_full_name.split("/")
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_id}/commits"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []
        commits = response.json()
        return [c["commit"]["message"] for c in commits]
    except Exception as e:
        print(f"PR {pr_id} Commit采集失败：{e}")
        return []

if __name__ == "__main__":
    pr_df = pd.read_csv("data/all_pr_basic.csv", encoding="utf-8-sig")
    all_code = []
    
    for _, row in tqdm(pr_df.iterrows(), total=len(pr_df), desc="采集代码变更数据"):
        repo, pr_id = row["repo"], row["pr_id"]
        files = get_pr_files(repo, pr_id)
        commits = get_pr_commits(repo, pr_id)
        
        all_code.append({
            "repo": repo,
            "pr_id": pr_id,
            "file_count": len(files),
            "modified_files": [f["filename"] for f in files],
            "modified_functions": list(set([func for f in files for func in f["modified_functions"]])),
            "total_additions": sum(f["additions"] for f in files),
            "total_deletions": sum(f["deletions"] for f in files),
            "commit_messages": commits
        })
        time.sleep(0.3)
    
    code_df = pd.DataFrame(all_code)
    code_df.to_csv("data/all_code_changes.csv", index=False, encoding="utf-8-sig")
    print(f"代码变更数据采集完成，共 {len(code_df)} 条，已保存到 data/all_code_changes.csv")