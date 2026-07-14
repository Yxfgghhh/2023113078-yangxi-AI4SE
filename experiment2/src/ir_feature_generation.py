import ast
import os
import re
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else None,
    "Accept": "application/vnd.github.v3+json"
}

LANGUAGE_KEYWORDS = re.compile(
    r"\b(if|else|elif|for|while|switch|case|try|except|finally|return|throw|break|continue|yield|await|async|function|def|class|interface|public|private|protected|static|var|let|const)\b",
    flags=re.IGNORECASE,
)


def safe_eval_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def get_github_pr_files(repo, pr_id):
    owner, repo_name = repo.split("/")
    url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_id}/files"
    files = []
    page = 1

    while True:
        params = {"per_page": 100, "page": page}
        response = requests.get(url, headers={k: v for k, v in HEADERS.items() if v}, params=params, timeout=30)
        if response.status_code == 403 and "rate limit" in response.text.lower():
            reset = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()) + 5, 5)
            print(f"触发限流，等待 {wait} 秒")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            print(f"PR {repo}#{pr_id} 文件获取失败，状态：{response.status_code}")
            break

        page_files = response.json()
        if not page_files:
            break
        files.extend(page_files)
        if len(page_files) < 100:
            break
        page += 1

    return files


def extract_added_lines(patch):
    if not isinstance(patch, str):
        return []
    added = []
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return added


def count_python_ast_nodes(code):
    if not code.strip():
        return 0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        wrapped = "def __copilot_temp():\n" + "\n".join("    " + l for l in code.splitlines())
        try:
            tree = ast.parse(wrapped)
        except SyntaxError:
            return max(0, len([l for l in code.splitlines() if l.strip()]))
    return len(list(ast.walk(tree)))


def heuristics_for_language(code):
    added_lines = [line for line in code.splitlines() if line.strip()]
    keyword_hits = LANGUAGE_KEYWORDS.findall(code)
    ast_count = max(1, len(added_lines) + len(keyword_hits) * 2)
    cfg_count = max(1, len(added_lines) + len(keyword_hits))
    return ast_count, cfg_count


def language_for_filename(filename):
    lower = filename.lower()
    if lower.endswith(".py"):
        return "python"
    if lower.endswith(('.js', '.jsx', '.ts', '.tsx')):
        return "javascript"
    if lower.endswith(('.java', '.kt', '.kts')):
        return "java"
    if lower.endswith(('.cpp', '.cc', '.c', '.h', '.hpp')):
        return "cpp"
    if lower.endswith(('.go',)):
        return "go"
    if lower.endswith(('.cs',)):
        return "csharp"
    return "other"


def compute_pr_ir_features(repo, pr_id):
    files = get_github_pr_files(repo, pr_id)
    total_ast = 0
    total_cfg = 0
    total_added = 0

    for file_info in files:
        patch = file_info.get("patch", "")
        added_lines = extract_added_lines(patch)
        total_added += len(added_lines)
        language = language_for_filename(file_info.get("filename", ""))
        added_code = "\n".join(added_lines)

        if language == "python":
            ast_count = count_python_ast_nodes(added_code)
            cfg_count = ast_count
        else:
            ast_count, cfg_count = heuristics_for_language(added_code)

        total_ast += ast_count
        total_cfg += cfg_count

    if total_added == 0:
        return {
            "ast_node_count": 0,
            "cfg_node_count": 0,
            "ast_density": 0.0,
            "cfg_density": 0.0,
        }

    return {
        "ast_node_count": total_ast,
        "cfg_node_count": total_cfg,
        "ast_density": total_ast / total_added,
        "cfg_density": total_cfg / total_added,
    }


def generate_ir_features():
    os.makedirs("results", exist_ok=True)
    clean_df = pd.read_csv("results/clean_pr_data.csv", encoding="utf-8-sig")
    records = []
    skipped = 0

    for _, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="生成 AST/CFG 特征"):
        repo = row["repo"]
        pr_id = int(row["pr_id"])
        try:
            features = compute_pr_ir_features(repo, pr_id)
        except Exception as exc:
            print(f"PR {repo}#{pr_id} 处理失败：{exc}")
            features = {"ast_node_count": 0, "cfg_node_count": 0, "ast_density": 0.0, "cfg_density": 0.0}
            skipped += 1

        records.append({
            "repo": repo,
            "pr_id": pr_id,
            **features,
        })
        time.sleep(0.2)

    result_df = pd.DataFrame(records)
    result_df.to_csv("results/pr_ast_cfg_features.csv", index=False, encoding="utf-8-sig")
    print(f"AST/CFG 特征生成完成, 记录数={len(result_df)}, 跳过={skipped}")
    return result_df


if __name__ == "__main__":
    generate_ir_features()
