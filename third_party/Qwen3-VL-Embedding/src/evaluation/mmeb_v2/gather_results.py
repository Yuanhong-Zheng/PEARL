#!/usr/bin/env python3
"""
Evaluation Results Collection Script
Collect and summarize multimodal evaluation task results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse


# Define task category configurations
TASK_CATEGORIES = {
    'IMG_CLS': {
        'metric': 'hit@1',
        'domain': 'image',
        'tasks': ['ImageNet-1K', 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 
                  'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211']
    },
    'IMG_QA': {
        'metric': 'hit@1',
        'domain': 'image',
        'tasks': ['OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA', 
                  'Visual7W', 'ScienceQA', 'VizWiz', 'GQA', 'TextVQA']
    },
    'IMG_RET': {
        'metric': 'hit@1',
        'domain': 'image',
        'tasks': ['MSCOCO_i2t', 'VisualNews_i2t', 'VisDial', 'MSCOCO_t2i', 
                  'VisualNews_t2i', 'WebQA', 'EDIS', 'Wiki-SS-NQ', 'CIRR', 
                  'NIGHTS', 'OVEN', 'FashionIQ']
    },
    'IMG_GRD': {
        'metric': 'hit@1',
        'domain': 'image',
        'tasks': ['MSCOCO', 'RefCOCO', 'RefCOCO-Matching', 'Visual7W-Pointing']
    },
    'VID_CLS': {
        'metric': 'hit@1',
        'domain': 'video',
        'tasks': ['SmthSmthV2', 'HMDB51', 'UCF101', 'K700', 'Breakfast']
    },
    'VID_QA': {
        'metric': 'hit@1',
        'domain': 'video',
        'tasks': ['Video-MME', 'NExTQA', 'EgoSchema', 'MVBench', 'ActivityNetQA']
    },
    'VID_RET': {
        'metric': 'hit@1',
        'domain': 'video',
        'tasks': ['MSR-VTT', 'MSVD', 'DiDeMo', 'YouCook2', 'VATEX']
    },
    'VID_MRET': {
        'metric': 'hit@1',
        'domain': 'video',
        'tasks': ['QVHighlight', 'Charades-STA', 'MomentSeeker']
    },
    'ViDoRe_v1': {
        'metric': 'ndcg_linear@5',
        'domain': 'visdoc',
        'tasks': ['ViDoRe_arxivqa', 'ViDoRe_docvqa', 'ViDoRe_infovqa', 
                  'ViDoRe_tabfquad', 'ViDoRe_tatdqa', 'ViDoRe_shiftproject',
                  'ViDoRe_syntheticDocQA_artificial_intelligence', 
                  'ViDoRe_syntheticDocQA_energy',
                  'ViDoRe_syntheticDocQA_government_reports', 
                  'ViDoRe_syntheticDocQA_healthcare_industry']
    },
    'ViDoRe_v2': {
        'metric': 'ndcg_linear@5',
        'domain': 'visdoc',
        'tasks': ['ViDoRe_esg_reports_human_labeled_v2', 
                  'ViDoRe_biomedical_lectures_v2_multilingual',
                  'ViDoRe_economics_reports_v2_multilingual', 
                  'ViDoRe_esg_reports_v2_multilingual']
    },
    'VisRag': {
        'metric': 'ndcg_linear@5',
        'domain': 'visdoc',
        'tasks': ['VisRAG_ArxivQA', 'VisRAG_ChartQA', 'VisRAG_MP-DocVQA', 
                  'VisRAG_SlideVQA', 'VisRAG_InfoVQA', 'VisRAG_PlotQA']
    },
    'OOD': {
        'metric': 'ndcg_linear@5',
        'domain': 'visdoc',
        'tasks': ['ViDoSeek-page', 'ViDoSeek-doc', 'MMLongBench-doc', 'MMLongBench-page']
    }
}


def load_score(eval_dir: Path, domain: str, task: str) -> Optional[Dict]:
    """Load score results for a single task"""
    score_file = eval_dir / domain / f"{task}_score.json"
    
    if not score_file.exists():
        print(f"Warning: File does not exist {score_file}")
        return None
    
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Failed to read file {score_file}: {e}")
        return None


def collect_results(eval_dir: Path) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Collect all evaluation results"""
    category_results = {}
    task_results = {}
    
    # Collect results for each category
    for category, config in TASK_CATEGORIES.items():
        metric = config['metric']
        domain = config['domain']
        tasks = config['tasks']
        
        scores = []
        for task in tasks:
            score_data = load_score(eval_dir, domain, task)
            if score_data and metric in score_data:
                score = score_data[metric] * 100  # Convert to percentage
                scores.append(score)
                task_results[task] = {metric: score}
            else:
                print(f"Warning: Task {task} missing metric {metric}")
        
        # Calculate average score for this category
        if scores:
            category_results[category] = sum(scores) / len(scores)
        else:
            category_results[category] = 0.0
            print(f"Warning: Category {category} has no valid results")
    
    return category_results, task_results


def compute_summary(category_results: Dict[str, float], task_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute summary results"""
    summary = category_results.copy()
    
    # Calculate averages for major categories based on all tasks
    img_categories = ['IMG_CLS', 'IMG_QA', 'IMG_RET', 'IMG_GRD']
    vid_categories = ['VID_CLS', 'VID_QA', 'VID_RET', 'VID_MRET']
    visdoc_categories = ['ViDoRe_v1', 'ViDoRe_v2', 'VisRag', 'OOD']
    
    # IMG average - average of all tasks in IMG categories
    img_scores = []
    for cat in img_categories:
        for task in TASK_CATEGORIES[cat]['tasks']:
            if task in task_results:
                metric_key = list(task_results[task].keys())[0]
                img_scores.append(task_results[task][metric_key])
    summary['IMG'] = sum(img_scores) / len(img_scores) if img_scores else 0.0
    
    # VID average - average of all tasks in VID categories
    vid_scores = []
    for cat in vid_categories:
        for task in TASK_CATEGORIES[cat]['tasks']:
            if task in task_results:
                metric_key = list(task_results[task].keys())[0]
                vid_scores.append(task_results[task][metric_key])
    summary['VID'] = sum(vid_scores) / len(vid_scores) if vid_scores else 0.0
    
    # Visdoc average - average of all tasks in Visdoc categories
    visdoc_scores = []
    for cat in visdoc_categories:
        for task in TASK_CATEGORIES[cat]['tasks']:
            if task in task_results:
                metric_key = list(task_results[task].keys())[0]
                visdoc_scores.append(task_results[task][metric_key])
    summary['Visdoc'] = sum(visdoc_scores) / len(visdoc_scores) if visdoc_scores else 0.0
    
    # ALL average - average of all tasks
    all_task_scores = []
    for task, scores in task_results.items():
        metric_key = list(scores.keys())[0]
        all_task_scores.append(scores[metric_key])
    summary['ALL'] = sum(all_task_scores) / len(all_task_scores) if all_task_scores else 0.0
    
    return summary


def print_table(headers: List[str], rows: List[List], title: str = "", max_width: int = 120):
    """Print table in terminal, split into multiple lines if too wide"""
    if title:
        print(f"\n{'='*max_width}")
        print(f"{title:^{max_width}}")
        print(f"{'='*max_width}")
    
    # Calculate maximum width for each column
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Split columns into chunks that fit within max_width
    chunks = []
    current_chunk = []
    current_width = 0
    
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        # Add 3 for " | " separator (or initial space)
        needed_width = width + (3 if current_chunk else 0)
        
        if current_width + needed_width > max_width and current_chunk:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = [i]
            current_width = width
        else:
            current_chunk.append(i)
            current_width += needed_width
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Print each chunk
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx > 0:
            print()  # Add blank line between chunks
        
        chunk_headers = [headers[i] for i in chunk]
        chunk_widths = [col_widths[i] for i in chunk]
        
        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(chunk_headers, chunk_widths))
        print(f"\n{header_line}")
        print("-" * len(header_line))
        
        # Print data rows
        for row in rows:
            chunk_cells = [row[i] for i in chunk]
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(chunk_cells, chunk_widths))
            print(row_line)
    
    print()


def save_tsv(file_path: Path, headers: List[str], rows: List[List]):
    """Save as TSV file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(headers) + '\n')
        for row in rows:
            f.write('\t'.join(str(cell) for cell in row) + '\n')
    print(f"Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect evaluation results')
    parser.add_argument('eval_dir', type=str, help='Evaluation results directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (optional)')
    parser.add_argument('--max_width', type=int, default=120, help='Maximum terminal width (default: 120)')
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory does not exist {eval_dir}")
        return
    
    # Collect results
    print("Collecting evaluation results...")
    category_results, task_results = collect_results(eval_dir)
    
    # Compute summary
    summary = compute_summary(category_results, task_results)
    
    # Prepare summary table data with task counts
    summary_categories = ['IMG_CLS', 'IMG_QA', 'IMG_RET', 'IMG_GRD', 'IMG', 
                          'VID_CLS', 'VID_QA', 'VID_RET', 'VID_MRET', 'VID',
                          'ViDoRe_v1', 'ViDoRe_v2', 'VisRag', 'OOD', 'Visdoc', 'ALL']
    
    # Calculate task counts for each category
    category_task_counts = {}
    for cat in summary_categories:
        if cat in TASK_CATEGORIES:
            category_task_counts[cat] = len(TASK_CATEGORIES[cat]['tasks'])
        elif cat == 'IMG':
            category_task_counts[cat] = sum(len(TASK_CATEGORIES[c]['tasks']) 
                                           for c in ['IMG_CLS', 'IMG_QA', 'IMG_RET', 'IMG_GRD'])
        elif cat == 'VID':
            category_task_counts[cat] = sum(len(TASK_CATEGORIES[c]['tasks']) 
                                           for c in ['VID_CLS', 'VID_QA', 'VID_RET', 'VID_MRET'])
        elif cat == 'Visdoc':
            category_task_counts[cat] = sum(len(TASK_CATEGORIES[c]['tasks']) 
                                           for c in ['ViDoRe_v1', 'ViDoRe_v2', 'VisRag', 'OOD'])
        elif cat == 'ALL':
            category_task_counts[cat] = sum(len(config['tasks']) for config in TASK_CATEGORIES.values())
    
    # Create headers with task counts
    summary_headers = [f"{cat}({category_task_counts.get(cat, 0)})" for cat in summary_categories]
    summary_row = [f"{summary.get(cat, 0):.1f}" for cat in summary_categories]
    
    # Prepare details table data
    all_tasks = []
    for config in TASK_CATEGORIES.values():
        all_tasks.extend(config['tasks'])
    
    details_headers = all_tasks
    details_row = []
    for task in all_tasks:
        if task in task_results:
            metric_key = list(task_results[task].keys())[0]
            score = task_results[task][metric_key]
            details_row.append(f"{score:.1f}")
        else:
            details_row.append("N/A")
    
    # Terminal output with adaptive width
    print_table(summary_headers, [summary_row], "Summary Results", max_width=args.max_width)
    print_table(details_headers, [details_row], "Detailed Results", max_width=args.max_width)
    
    # File output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_tsv(output_dir / 'summary.tsv', summary_headers, [summary_row])
        save_tsv(output_dir / 'details.tsv', details_headers, [details_row])
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()