import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import sacrebleu
    from rouge_score import rouge_scorer
except Exception:
    sacrebleu = None
    rouge_scorer = None


RESULTS_GLOB = os.path.join("results", "llm_review_results_ctx*_pt*.csv")
ALL_REVIEWS = os.path.join("results", "all_pr_reviews.csv")
OUTPUT_METRICS = os.path.join("results", "llm_evaluation_metrics.csv")
ANALYSIS_TXT = os.path.join("results", "llm_metrics_comparison_analysis.txt")


def load_reference_comments():
    if not os.path.exists(ALL_REVIEWS):
        return pd.DataFrame(columns=["repo", "pr_id", "reference_comment"]) 
    r = pd.read_csv(ALL_REVIEWS, encoding="utf-8-sig")
    r = r.fillna("")
    grouped = r.groupby(["repo", "pr_id"])['review_comment'].apply(lambda x: "\n\n".join([s for s in x if s])).reset_index()
    grouped = grouped.rename(columns={'review_comment': 'reference_comment'})
    return grouped


def score_file(path, refs_df):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return None

    # merge reference comments
    merged = df.merge(refs_df, on=["repo", "pr_id"], how="left")
    # Merge prediction metrics
    valid_pred = merged[merged['pred_merge'].notna() & merged['merged'].notna()].copy()
    y_true = valid_pred['merged'].astype(int)
    y_pred = valid_pred['pred_merge'].astype(int)

    if len(y_true) > 0:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        acc = prec = rec = f1 = np.nan

    # Text metrics (BLEU / ROUGE-L) comparing generated response to aggregated human reviews
    gens = merged['response'].fillna("").astype(str).tolist()
    refs = merged['reference_comment'].fillna("").astype(str).tolist()

    bleu = np.nan
    rouge_l = np.nan
    if sacrebleu is not None and any(r.strip() for r in refs):
        try:
            # sacrebleu expects list of hypothesis and list of reference lists
            bleu = float(sacrebleu.corpus_bleu(gens, [refs]).score)
        except Exception:
            bleu = np.nan

    if rouge_scorer is not None and any(r.strip() for r in refs):
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = [scorer.score(r, g) for r, g in zip(refs, gens)]
            # average fmeasure
            rouge_l = float(np.nanmean([s['rougeL'].fmeasure for s in scores]))
        except Exception:
            rouge_l = np.nan

    avg_time = float(merged['elapsed_s'].dropna().astype(float).mean()) if 'elapsed_s' in merged.columns else np.nan

    # parse context and prompt from filename
    base = os.path.basename(path)
    # expected pattern: llm_review_results_ctx{n}_pt{type}.csv (maybe with prefix)
    ctx = None
    pt = None
    m = None
    try:
        parts = base.replace('.csv', '').split('_')
        for p in parts:
            if p.startswith('ctx'):
                ctx = int(p.replace('ctx', ''))
            if p.startswith('pt'):
                pt = p.replace('pt', '')
    except Exception:
        pass

    return {
        'file': path,
        'context_type': ctx,
        'prompt_type': pt,
        'n_samples': len(merged),
        'n_with_ref': int(sum(1 for r in refs if r.strip())),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'bleu': bleu,
        'rouge_l_fmeasure': rouge_l,
        'avg_inference_time_s': avg_time,
    }


def main():
    refs = load_reference_comments()
    files = sorted(glob.glob(RESULTS_GLOB))
    metrics = []
    for f in tqdm(files, desc='Scoring'):
        r = score_file(f, refs)
        if r:
            metrics.append(r)

    if metrics:
        dfm = pd.DataFrame(metrics)
        dfm.to_csv(OUTPUT_METRICS, index=False, encoding='utf-8-sig')
        print(f"Saved evaluation metrics to {OUTPUT_METRICS}")
    else:
        print("No result files scored.")

    # Simple comparison with existing ai_model_metrics.csv if present
    ai_metrics_path = os.path.join('results', 'ai_model_metrics.csv')
    with open(ANALYSIS_TXT, 'w', encoding='utf-8') as fh:
        if os.path.exists(ai_metrics_path) and metrics:
            try:
                aim = pd.read_csv(ai_metrics_path, encoding='utf-8-sig')
                fh.write('Comparison with results/ai_model_metrics.csv\n')
                fh.write('Loaded ai_model_metrics.csv with %d rows\n\n' % len(aim))
                # show top by f1 from our df
                top = dfm.sort_values('f1', ascending=False).head(5)
                fh.write('Top 5 LLM runs by F1:\n')
                for _, row in top.iterrows():
                    fh.write(f"{row['file']}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, BLEU={row['bleu']}, ROUGE_L={row['rouge_l_fmeasure']}\n")
                fh.write('\nAI baseline metrics head (first 5 rows):\n')
                fh.write(aim.head().to_string(index=False))
            except Exception as e:
                fh.write(f"Failed to load ai_model_metrics.csv: {e}\n")
        else:
            fh.write('No ai_model_metrics.csv found or no metrics computed.\n')
    print(f"Wrote analysis to {ANALYSIS_TXT}")


if __name__ == '__main__':
    main()
