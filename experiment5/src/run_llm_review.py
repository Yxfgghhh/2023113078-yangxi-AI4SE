import ast
import json
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro").strip()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

DEFAULT_MAX_ROWS = 80
DEFAULT_MAX_TOKENS = 1024


def safe_eval_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def load_cleaned_datasets(clean_path="results/clean_pr_data.csv"):
    os.makedirs("results", exist_ok=True)
    changes_path = "results/all_code_changes.csv"

    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Missing cleaned dataset: {clean_path}. Run syc.data_cleaning.clean_data() first.")
    if not os.path.exists(changes_path):
        raise FileNotFoundError(f"Missing code change dataset: {changes_path}. Ensure results/all_code_changes.csv exists.")

    clean_df = pd.read_csv(clean_path, encoding="utf-8-sig")
    change_df = pd.read_csv(changes_path, encoding="utf-8-sig")

    clean_df["modified_files"] = clean_df["modified_files"].apply(safe_eval_list)
    clean_df["commit_messages"] = clean_df["commit_messages"].apply(safe_eval_list)
    change_df["modified_files"] = change_df["modified_files"].apply(safe_eval_list)
    change_df["commit_messages"] = change_df["commit_messages"].apply(safe_eval_list)

    merged = clean_df.merge(change_df, on=["repo", "pr_id"], how="left", suffixes=("", "_code"))
    merged["title"] = merged["title"].fillna("")
    merged["body"] = merged["body"].fillna("")
    merged["modified_files"] = merged["modified_files"].apply(lambda x: x if isinstance(x, list) else [])
    merged["commit_messages"] = merged["commit_messages"].apply(lambda x: x if isinstance(x, list) else [])
    merged["diff_summary"] = merged.apply(generate_diff_summary, axis=1)

    return merged


def fetch_diff_texts(df: pd.DataFrame, max_prs: int = 50):
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is required to fetch actual PR patch diffs from GitHub.")

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    results = []
    for _, row in tqdm(df.head(max_prs).iterrows(), total=min(len(df), max_prs), desc="Fetching diffs"):
        repo = row["repo"]
        pr_id = row["pr_id"]
        owner, repo_name = repo.split("/")
        url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_id}/files"
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            files = response.json()
            patch_sections = []
            for file_info in files:
                patch = file_info.get("patch")
                filename = file_info.get("filename")
                if patch:
                    patch_sections.append(f"File: {filename}\n{patch}")
            diff_text = "\n\n".join(patch_sections) if patch_sections else row.get("diff_summary", "")
        except Exception as exc:
            diff_text = row.get("diff_summary", "") + f"\n\n[Diff fetch failed: {exc}]"
        results.append(((repo, pr_id), diff_text))

    diff_map = {key: text for key, text in results}
    df = df.copy()
    df["diff_text"] = df.apply(lambda row: diff_map.get((row["repo"], row["pr_id"]), row.get("diff_summary", "")), axis=1)
    return df


def prepare_review_dataset(fetch_diff: bool = False, fetch_limit: int = 50, clean_path: str = "results/clean_pr_data.csv"):
    df = load_cleaned_datasets(clean_path=clean_path)
    if fetch_diff:
        print(f"Fetching up to {fetch_limit} PR patch diffs from GitHub...")
        df = fetch_diff_texts(df, max_prs=fetch_limit)
    return df


def generate_diff_summary(row):
    # 为缺失的 Diff 提供一个简要替代描述，优先保留修改文件和行数信息
    files = row.get("modified_files") or []
    additions = row.get("total_additions")
    deletions = row.get("total_deletions")
    if isinstance(additions, float) and pd.isna(additions):
        additions = None
    if isinstance(deletions, float) and pd.isna(deletions):
        deletions = None

    file_text = "\n".join([f"- {f}" for f in files[:20]]) or "(no modified files listed)"
    summary = ["Diff summary:", f"Modified files ({len(files)}):", file_text]
    if additions is not None or deletions is not None:
        summary.append(f"Additions: {additions if additions is not None else 'unknown'}, Deletions: {deletions if deletions is not None else 'unknown'}")

    summary.append("\nNote: the dataset does not contain actual unified diff patches, so this summary uses file-level change metadata.")
    return "\n".join(summary)


def build_context(row, context_type: int):
    parts = []
    diff_text = row.get("diff_text") or row.get("diff_summary", "")
    if context_type in (1, 2, 3, 4):
        parts.append(diff_text)

    if context_type in (2, 4):
        body = row.get("body", "")
        if body:
            parts.append("PR Description:\n" + body)

    if context_type in (3, 4):
        commits = row.get("commit_messages", [])
        if commits:
            parts.append("Commit Messages:\n" + "\n---\n".join(str(m) for m in commits[:5]))

    if context_type == 4:
        modified_files = row.get("modified_files", [])
        if modified_files:
            parts.append("Modified Files:\n" + "\n".join(f"- {f}" for f in modified_files[:20]))

    return "\n\n".join(parts)


def truncate_text(text: str, max_chars: int = 4000):
    if not isinstance(text, str):
        return ""
    return text if len(text) <= max_chars else text[:max_chars - 300] + "\n... [truncated]"


def build_prompt_text(context: str, prompt_type: str):
    base_instructions = (
        "You are a senior GitHub code reviewer.\n"
        "Based on the provided pull request context, answer the following:\n"
        "1. Merge Prediction: should this PR be merged? Answer Yes or No.\n"
        "2. Review Comment: provide a concise, practical code review comment that explains strengths, issues, and improvement suggestions.\n"
        "Include your reasoning in the response."
    )

    prompt_type = prompt_type.lower()
    if prompt_type == "zero":
        prompt = (
            base_instructions
            + "\nUse only the context below and do not invent extra facts.\n\nContext:\n" + context
        )
    elif prompt_type == "few":
        examples = get_few_shot_examples()
        prompt = (
            "You are a senior GitHub reviewer. Learn from the examples and apply the same style.\n"
            + "For the new pull request, provide Merge Prediction and Review Comment.\n\n"
            + "Examples:\n" + examples + "\n\nNew PR Context:\n" + context
        )
    elif prompt_type == "cot":
        prompt = (
            base_instructions
            + "\nThink step by step and analyze the PR context before giving your answer.\n\nContext:\n" + context
        )
    elif prompt_type == "role":
        prompt = (
            "You are a meticulous senior engineer and reviewer.\n"
            "Review the PR carefully, then provide a merge decision and targeted review comments.\n\nContext:\n"
            + context
        )
    else:
        prompt = base_instructions + "\nContext:\n" + context

    return truncate_text(prompt, 6000)


def get_few_shot_examples():
    examples = [
        {
            "context": (
                "Diff summary:\n- src/app.py\n- tests/test_app.py\nAdditions: 15, Deletions: 2\n\n"
                "PR Description:\nFixes a null pointer exception in the request handler.\n\n"
                "Commit Messages:\nfix: prevent null pointer in request handler\n"
                "Modified Files:\n- src/app.py\n- tests/test_app.py"
            ),
            "response": (
                "Merge Prediction: Yes.\n"
                "Review Comment: The change appears correct and addresses the null pointer issue.\n"
                "Please add a unit test for the new request handler branch and confirm the error path is covered."
            )
        },
        {
            "context": (
                "Diff summary:\n- src/auth.py\n- src/session.py\nAdditions: 30, Deletions: 5\n\n"
                "PR Description:\nImproves authentication state handling.\n\n"
                "Commit Messages:\nrefactor: simplify auth token refresh\n"
                "Modified Files:\n- src/auth.py\n- src/session.py"
            ),
            "response": (
                "Merge Prediction: No.\n"
                "Review Comment: The auth-state changes are promising, but the new error-handling branch is not covered by tests and may break session refresh.\n"
                "Please verify the token expiry behavior and add regression tests for refresh failure cases."
            )
        }
    ]
    return "\n\n".join(
        f"Context:\n{item['context']}\nResponse:\n{item['response']}" for item in examples
    )


def init_llm_client():
    if LLM_PROVIDER != "openai":
        raise RuntimeError("Only OpenAI-compatible provider is supported in this script.")
    if OpenAI is None:
        raise ImportError("The openai package is required to run LLM inference.")
    if not LLM_API_KEY or not LLM_API_BASE:
        raise RuntimeError("LLM_API_KEY and LLM_API_BASE must be set in environment or .env.")
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)


def call_llm(client, prompt_text: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.2):
    start = time.time()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a senior GitHub code reviewer."},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        n=1,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content.strip()
    return text, elapsed


def extract_merge_prediction(response_text: str):
    if not isinstance(response_text, str):
        return None
    text_lower = response_text.lower()
    if re.search(r'\b(no|reject|do not merge|should not merge|no\.?)\b', text_lower):
        return 0
    if re.search(r'\b(yes|approve|merge|should merge|merge it)\b', text_lower):
        return 1
    if "merge prediction" in text_lower or "merge:" in text_lower:
        if re.search(r'merge.*:\s*(yes|y|approved|1)', text_lower):
            return 1
        if re.search(r'merge.*:\s*(no|n|rejected|0)', text_lower):
            return 0
    return None


def run_review(limit: int = DEFAULT_MAX_ROWS, dry_run: bool = False, skip_existing: bool = False, fetch_diff: bool = False, fetch_limit: int = 50, clean_path: str = "results/clean_pr_data.csv", output_prefix: str = ""):
    df = prepare_review_dataset(fetch_diff=fetch_diff, fetch_limit=fetch_limit, clean_path=clean_path)
    if limit is not None:
        df = df.head(limit)

    context_types = [1, 2, 3, 4]
    prompt_types = ["zero", "few", "cot", "role"]

    if dry_run:
        print("Dry run mode: building prompts without calling the LLM.")

    if not dry_run:
        client = init_llm_client()
    else:
        client = None

    results = []
    df = prepare_review_dataset(fetch_diff=fetch_diff, fetch_limit=fetch_limit, clean_path=clean_path)
    if limit is not None:
        df = df.head(limit)

    for context_type in context_types:
        for prompt_type in prompt_types:
            output_path = f"results/{output_prefix}llm_review_results_ctx{context_type}_pt{prompt_type}.csv"
            if skip_existing and os.path.exists(output_path):
                print(f"Skipping existing file: {output_path}")
                continue

            rows = []
            print(f"Processing context={context_type}, prompt={prompt_type}, rows={len(df)}")
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ctx{context_type}_pt{prompt_type}"):
                context = build_context(row, context_type)
                prompt_text = build_prompt_text(context, prompt_type)
                response_text = None
                elapsed = None
                if dry_run:
                    response_text = "[DRY_RUN]"
                    elapsed = 0.0
                else:
                    try:
                        response_text, elapsed = call_llm(client, prompt_text)
                    except Exception as exc:
                        response_text = f"LLM_ERROR: {exc}"
                        elapsed = 0.0

                rows.append({
                    "repo": row["repo"],
                    "pr_id": row["pr_id"],
                    "title": row.get("title", ""),
                    "merged": row.get("merged"),
                    "context_type": context_type,
                    "prompt_type": prompt_type,
                    "prompt": prompt_text,
                    "response": response_text,
                    "pred_merge": extract_merge_prediction(response_text),
                    "elapsed_s": elapsed,
                    "response_length": len(response_text) if isinstance(response_text, str) else 0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved {len(rows)} review rows to {output_path}")
            results.extend(rows)

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-based PR merge prediction and review comment generation.")
    parser.add_argument("--limit", type=int, default=DEFAULT_MAX_ROWS, help="Maximum PR rows to evaluate.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without calling the LLM.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--fetch-diff", action="store_true", help="Fetch actual PR patch diffs from GitHub when possible.")
    parser.add_argument("--fetch-limit", type=int, default=50, help="Maximum number of PRs to fetch patch diffs for.")
    parser.add_argument("--clean-path", default="results/clean_pr_data.csv", help="Path to the cleaned PR dataset to evaluate.")
    parser.add_argument("--output-prefix", default="", help="Filename prefix for generated LLM result CSV files.")
    args = parser.parse_args()

    if args.force:
        args.skip_existing = False

    if args.force:
        for context_type in [1, 2, 3, 4]:
            for prompt_type in ["zero", "few", "cot", "role"]:
                out = f"results/{args.output_prefix}llm_review_results_ctx{context_type}_pt{prompt_type}.csv"
                if os.path.exists(out):
                    os.remove(out)

    run_review(
        limit=args.limit,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        fetch_diff=args.fetch_diff,
        fetch_limit=args.fetch_limit,
        clean_path=args.clean_path,
        output_prefix=args.output_prefix,
    )
