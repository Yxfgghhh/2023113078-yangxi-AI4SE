import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict

try:
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def safe_eval_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        import ast
        return ast.literal_eval(value)
    except Exception:
        return []


def extract_merge_prediction(response_text):
    """从LLM响应中提取是否应该Merge的预测"""
    if not isinstance(response_text, str) or response_text == 'LLM_NOT_CONFIGURED':
        return None
    
    text_lower = response_text.lower()
    
    # 查找明确的Yes/No
    if re.search(r'\b(yes|approve|merge|should merge)\b', text_lower):
        return 1
    if re.search(r'\b(no|reject|do not merge|should not merge)\b', text_lower):
        return 0
    
    # 查找Merge prediction字段
    if 'merge prediction' in text_lower or 'merge:' in text_lower:
        if re.search(r':\s*(yes|y|approved|1)', text_lower):
            return 1
        if re.search(r':\s*(no|n|rejected|0)', text_lower):
            return 0
    
    return None


def calculate_response_quality(response_text):
    """计算响应文本的基本质量指标"""
    if not isinstance(response_text, str) or response_text == 'LLM_NOT_CONFIGURED':
        return {
            'length': 0,
            'has_merge_pred': 0,
            'has_comments': 0,
        }
    
    length = len(response_text)
    has_merge = 1 if extract_merge_prediction(response_text) is not None else 0
    has_comments = 1 if len(response_text) > 100 else 0
    
    return {
        'length': length,
        'has_merge_pred': has_merge,
        'has_comments': has_comments,
    }


def compute_metrics_per_file(csv_path, ground_truth_df):
    """计算单个CSV文件的指标"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 合并ground truth
    df = df.merge(ground_truth_df[['repo', 'pr_id', 'merged']], on=['repo', 'pr_id'], how='left')
    
    # 提取merge预测
    df['pred_merge'] = df['response'].apply(extract_merge_prediction)
    
    # 计算质量指标
    quality = df['response'].apply(calculate_response_quality)
    for key in ['length', 'has_merge_pred', 'has_comments']:
        df[key] = quality.apply(lambda x: x[key])
    
    # 只有pred_merge非空的记录才参与评估
    valid_mask = df['pred_merge'].notna()
    valid_df = df[valid_mask]
    
    if len(valid_df) == 0:
        return {
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan,
            'avg_resp_length': df['length'].mean(),
            'merge_pred_rate': 0,
            'avg_elapsed': df['elapsed_s'].mean(),
            'count': len(df),
        }
    
    y_true = valid_df['merged'].values.astype(float)
    y_pred = valid_df['pred_merge'].values.astype(float)
    
    # 计算指标
    accuracy = (y_true == y_pred).mean()
    
    # Precision、Recall、F1
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_resp_length': df['length'].mean(),
        'merge_pred_rate': (df['pred_merge'] == 1).sum() / len(df) if len(df) > 0 else 0,
        'avg_elapsed': df['elapsed_s'].mean(),
        'count': len(df),
        'valid_count': len(valid_df),
    }


def analyze_all_results(clean_path='results/clean_pr_data.csv', results_dir='results', prefix='llm_review_results_', output_path='results/llm_performance_analysis.csv'):
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载ground truth
    clean_df = pd.read_csv(clean_path, encoding='utf-8-sig')
    ground_truth = clean_df[['repo', 'pr_id', 'merged']].copy()
    
    # 扫描所有llm_review_results文件
    results = []
    
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith(prefix) and filename.endswith('.csv'):
            filepath = os.path.join(results_dir, filename)
            
            # 解析文件名
            match = re.match(r'llm_review_results_ctx(\d)_pt(\w+)\.csv', filename)
            if not match:
                continue
            
            ctx_type = int(match.group(1))
            prompt_type = match.group(2)
            
            print(f'Analyzing {filename}...')
            metrics = compute_metrics_per_file(filepath, ground_truth)
            
            result = {
                'context_type': ctx_type,
                'prompt_type': prompt_type,
                'filename': filename,
                **metrics
            }
            results.append(result)
    
    # 生成结果表
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['context_type', 'prompt_type'])
    
    # 保存到CSV
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'\nPerformance analysis saved to {output_path}')
    
    # 打印摘要
    print('\n=== LLM Performance Summary ===')
    print(results_df.to_string(index=False))
    
    # 按上下文对比
    print('\n=== By Context Type ===')
    by_context = results_df.groupby('context_type')[['accuracy', 'precision', 'recall', 'f1', 'avg_resp_length', 'avg_elapsed']].mean()
    print(by_context)
    
    # 按Prompt对比
    print('\n=== By Prompt Type ===')
    by_prompt = results_df.groupby('prompt_type')[['accuracy', 'precision', 'recall', 'f1', 'avg_resp_length', 'avg_elapsed']].mean()
    print(by_prompt)
    
    # 找最优配置
    print('\n=== Best Configurations ===')
    print(f'Best F1-score: {results_df.loc[results_df["f1"].idxmax()][["context_type", "prompt_type", "f1", "accuracy", "precision", "recall"]].to_dict()}')
    print(f'Fastest: {results_df.loc[results_df["avg_elapsed"].idxmin()][["context_type", "prompt_type", "avg_elapsed"]].to_dict()}')
    print(f'Longest response: {results_df.loc[results_df["avg_resp_length"].idxmax()][["context_type", "prompt_type", "avg_resp_length"]].to_dict()}')
    
    return results_df


if __name__ == '__main__':
    analyze_all_results()
