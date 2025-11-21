#!/usr/bin/env python3
"""
Annotation Evaluation Script for Secret Hitler Player Project

This script evaluates multi-class classification performance of annotation models
against ground truth data. It supports multi-label classification where each text
can have multiple annotation categories.

Ground truth is in: replay_data_ann/ and llama-ann/
Model predictions are in other folders (excluding web/)

Categories: Authority, Consistency, Friendship/Liking, Reciprocation, Scarcity, Social Validation
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class AnnotationEvaluator:
    def __init__(self, annotation_dir: str, ground_truth_dirs: List[str] = None):
        self.annotation_dir = Path(annotation_dir)
        self.categories = [
            'Authority', 'Consistency', 'Friendship/Liking', 
            'Reciprocation', 'Scarcity', 'Social Validation'
        ]
        self.ground_truth_dirs = ground_truth_dirs or ['replay_data_ann', 'llama-ann']
        self.exclude_dirs = ['web', 'src']
        
    def load_annotations(self, dir_path: Path) -> List[Dict]:
        """Load all annotation files from a directory."""
        annotations = []
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist")
            return annotations
            
        for file_path in dir_path.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    annotations.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return annotations
    
    def get_model_dirs(self) -> List[str]:
        """Get all model directories (excluding ground truth and web)."""
        all_dirs = [d.name for d in self.annotation_dir.iterdir() if d.is_dir()]
        model_dirs = [d for d in all_dirs 
                     if d not in self.ground_truth_dirs + self.exclude_dirs]
        return sorted(model_dirs)
    
    def annotations_to_binary_matrix(self, annotations: List[Dict]) -> List[List[int]]:
        """Convert annotations to binary matrix format for multi-label evaluation."""
        matrix = [[0] * len(self.categories) for _ in range(len(annotations))]
        
        for i, item in enumerate(annotations):
            item_cats = item.get('annotation', [])
            for cat in item_cats:
                if cat in self.categories:
                    cat_idx = self.categories.index(cat)
                    matrix[i][cat_idx] = 1
        
        return matrix
    
    def match_annotations(self, gt_annotations: List[Dict], pred_annotations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Match ground truth and prediction annotations by text."""
        # Create lookup dictionary for predictions
        pred_lookup = {item['text']: item for item in pred_annotations}
        
        matched_gt = []
        matched_pred = []
        
        for gt_item in gt_annotations:
            text = gt_item['text']
            if text in pred_lookup:
                matched_gt.append(gt_item)
                matched_pred.append(pred_lookup[text])
        
        return matched_gt, matched_pred
    
    def evaluate_model(self, model_name: str) -> Dict:
        """Evaluate a single model against ground truth."""
        print(f"\nEvaluating model: {model_name}")
        
        # Load ground truth
        gt_annotations = []
        for gt_dir in self.ground_truth_dirs:
            gt_path = self.annotation_dir / gt_dir
            gt_annotations.extend(self.load_annotations(gt_path))
        
        # Load model predictions
        model_path = self.annotation_dir / model_name
        pred_annotations = self.load_annotations(model_path)
        
        if not gt_annotations:
            print(f"No ground truth annotations found")
            return {}
        
        if not pred_annotations:
            print(f"No predictions found for {model_name}")
            return {}
        
        # Match annotations by text
        matched_gt, matched_pred = self.match_annotations(gt_annotations, pred_annotations)
        
        if not matched_gt:
            print(f"No matching texts found between ground truth and {model_name}")
            return {}
        
        print(f"Matched {len(matched_gt)} examples")
        
        # Convert to binary matrices
        gt_matrix = self.annotations_to_binary_matrix(matched_gt)
        pred_matrix = self.annotations_to_binary_matrix(matched_pred)
        
        # Calculate metrics
        results = self.calculate_basic_metrics(gt_matrix, pred_matrix, model_name)
        
        return results
    
    def calculate_basic_metrics(self, y_true: List[List[int]], y_pred: List[List[int]], model_name: str) -> Dict:
        """Calculate basic evaluation metrics without external dependencies."""
        results = {'model': model_name}
        
        if not y_true or not y_pred:
            return results
        
        # Exact match accuracy
        exact_matches = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        results['exact_match_accuracy'] = exact_matches / len(y_true)
        
        # Per-category metrics
        category_results = {}
        for cat_idx, category in enumerate(self.categories):
            # True positives, false positives, false negatives
            tp = sum(1 for i in range(len(y_true)) if y_true[i][cat_idx] == 1 and y_pred[i][cat_idx] == 1)
            fp = sum(1 for i in range(len(y_true)) if y_true[i][cat_idx] == 0 and y_pred[i][cat_idx] == 1)
            fn = sum(1 for i in range(len(y_true)) if y_true[i][cat_idx] == 1 and y_pred[i][cat_idx] == 0)
            tn = sum(1 for i in range(len(y_true)) if y_true[i][cat_idx] == 0 and y_pred[i][cat_idx] == 0)
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Support (number of true instances)
            support = tp + fn
            
            category_results[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
            
            results[f'{category}_precision'] = precision
            results[f'{category}_recall'] = recall
            results[f'{category}_f1'] = f1
            results[f'{category}_support'] = support
        
        # Macro averages
        precisions = [category_results[cat]['precision'] for cat in self.categories]
        recalls = [category_results[cat]['recall'] for cat in self.categories]
        f1s = [category_results[cat]['f1'] for cat in self.categories]
        
        results['macro_precision'] = sum(precisions) / len(precisions)
        results['macro_recall'] = sum(recalls) / len(recalls)
        results['macro_f1'] = sum(f1s) / len(f1s)
        
        # Micro averages
        total_tp = sum(category_results[cat]['tp'] for cat in self.categories)
        total_fp = sum(category_results[cat]['fp'] for cat in self.categories)
        total_fn = sum(category_results[cat]['fn'] for cat in self.categories)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        results['micro_precision'] = micro_precision
        results['micro_recall'] = micro_recall
        results['micro_f1'] = micro_f1
        
        # Hamming loss (fraction of wrong labels)
        total_labels = len(y_true) * len(self.categories)
        wrong_labels = sum(sum(1 for j in range(len(self.categories)) if y_true[i][j] != y_pred[i][j]) 
                          for i in range(len(y_true)))
        results['hamming_loss'] = wrong_labels / total_labels
        
        # Class distribution
        for cat_idx, category in enumerate(self.categories):
            gt_count = sum(y_true[i][cat_idx] for i in range(len(y_true)))
            pred_count = sum(y_pred[i][cat_idx] for i in range(len(y_pred)))
            results[f'{category}_gt_count'] = gt_count
            results[f'{category}_pred_count'] = pred_count
        
        return results
    
    def run_evaluation(self, models: List[str] = None, save_results: bool = True, filename_suffix: str = "") -> List[Dict]:
        """Run evaluation for all or specified models."""
        if models is None:
            models = self.get_model_dirs()
        
        print(f"Found model directories: {models}")
        print(f"Ground truth directories: {self.ground_truth_dirs}")
        
        all_results = []
        
        for model in models:
            results = self.evaluate_model(model)
            if results:
                all_results.append(results)
        
        if not all_results:
            print("No results to display")
            return []
        
        if save_results:
            # Save results as JSON with suffix
            filename = f'evaluation_results{filename_suffix}.json'
            with open(self.annotation_dir / filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\nResults saved to: {filename}")
        
        return all_results
    
    def print_detailed_table(self, all_results: List[Dict]):
        """Print detailed comparison table with per-category metrics."""
        if not all_results:
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE (Overall Metrics)")
        print("="*80)
        
        # Print header
        header = f"{'Model':<25} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Hamming':<8}"
        print(f"\n{header}")
        print("-" * 57)
        
        # Sort by F1
        sorted_results = sorted(all_results, key=lambda x: x.get('macro_f1', 0), reverse=True)
        
        for result in sorted_results:
            model = result.get('model', 'Unknown')[:24]
            f1 = result.get('macro_f1', 0)
            prec = result.get('macro_precision', 0)
            rec = result.get('macro_recall', 0)
            hamming = result.get('hamming_loss', 0)
            print(f"{model:<25} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {hamming:<8.4f}")
        
        print(f"\nNote: Hamming Loss = fraction of incorrectly predicted labels (lower is better)")
        
        # Per-category detailed table
        print("\n" + "="*80)
        print("PER-CATEGORY METRICS")
        print("="*80)
        
        for result in sorted_results:
            model_name = result.get('model', 'Unknown')
            print(f"\n{model_name}:")
            print(f"{'Category':<20} {'F1':<8} {'Prec':<10} {'Rec':<8} {'Support':<8}")
            print("-" * 54)
            
            for category in self.categories:
                f1 = result.get(f'{category}_f1', 0)
                prec = result.get(f'{category}_precision', 0)
                rec = result.get(f'{category}_recall', 0)
                support = result.get(f'{category}_support', 0)
                print(f"{category:<20} {f1:<8.4f} {prec:<10.4f} {rec:<8.4f} {support:<8}")
        
        # Condensed per-category F1 comparison table
        print("\n" + "="*80)
        print("PER-CATEGORY F1 COMPARISON")
        print("="*80)
        
        # Build header with shortened category names
        cat_abbrev = [cat[:6] for cat in self.categories]
        header = f"\n{'Model':<25} " + ' '.join([f"{abbr:<8}" for abbr in cat_abbrev])
        print(header)
        print("-" * (25 + 8 * len(self.categories) + len(self.categories)))
        
        for result in sorted_results:
            model = result.get('model', 'Unknown')[:24]
            cat_scores = ' '.join([f"{result.get(f'{cat}_f1', 0):<8.4f}" for cat in self.categories])
            print(f"{model:<25} {cat_scores}")
    
    def print_summary(self, all_results: List[Dict]):
        """Print evaluation summary."""
        if not all_results:
            return
        
        print("\n" + "="*80)
        print("ANNOTATION EVALUATION SUMMARY")
        print("="*80)
        
        # Overall performance ranking
        sorted_results = sorted(all_results, key=lambda x: x.get('macro_f1', 0), reverse=True)
        
        print(f"\nTop Models by Macro F1 Score:")
        print("-" * 50)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i:2d}. {result['model']:<30} F1: {result.get('macro_f1', 0):.4f}")

        print(f"\nTop Models by Hamming Loss (lower is better):")
        print("-" * 50)
        for i, result in enumerate(sorted(all_results, key=lambda x: x.get('hamming_loss', 1))[:5], 1):
            print(f"{i:2d}. {result['model']:<30} Hamming Loss: {result.get('hamming_loss', 1):.4f}")
        
        if sorted_results:
            print(f"\nDetailed Metrics for Top Model ({sorted_results[0]['model']}):")
            print("-" * 60)
            top_model = sorted_results[0]
            
            metrics = ['exact_match_accuracy', 'hamming_loss', 
                      'macro_precision', 'macro_recall', 'macro_f1',
                      'micro_precision', 'micro_recall', 'micro_f1']
            for metric in metrics:
                if metric in top_model:
                    print(f"{metric:<25}: {top_model[metric]:.4f}")
            
            print(f"\nPer-Category F1 Scores (Top Model):")
            print("-" * 40)
            for category in self.categories:
                f1_col = f'{category}_f1'
                if f1_col in top_model:
                    print(f"{category:<20}: {top_model[f1_col]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate annotation models')
    parser.add_argument('--annotation-dir', '-d', 
                       default='/home/bauer/sh_test/secret-hitler-player/annotation',
                       help='Path to annotation directory')
    parser.add_argument('--models', '-m', nargs='+', 
                       help='Specific models to evaluate (default: all)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--ground-truth', '-g', choices=['all', 'llama-ann', 'replay_data_ann', 'separate'],
                       default='all',
                       help='Which ground truth to evaluate against: all (both combined), llama-ann only, replay_data_ann only, or separate (run both separately)')
    
    args = parser.parse_args()
    
    if args.ground_truth == 'separate':
        # Run separate evaluations for each ground truth dataset
        print("Running separate evaluations for each ground truth dataset...\n")
        
        # Evaluate against llama-ann only
        print("="*60)
        print("EVALUATING AGAINST LLAMA-ANN ONLY")
        print("="*60)
        evaluator_llama = AnnotationEvaluator(args.annotation_dir, ['llama-ann'])
        results_llama = evaluator_llama.run_evaluation(
            models=args.models, 
            save_results=not args.no_save,
            filename_suffix='_llama_ann'
        )
        evaluator_llama.print_detailed_table(results_llama)
        evaluator_llama.print_summary(results_llama)
        
        # Evaluate against replay_data_ann only
        print("\n" + "="*60)
        print("EVALUATING AGAINST REPLAY_DATA_ANN ONLY")
        print("="*60)
        evaluator_replay = AnnotationEvaluator(args.annotation_dir, ['replay_data_ann'])
        results_replay = evaluator_replay.run_evaluation(
            models=args.models, 
            save_results=not args.no_save,
            filename_suffix='_replay_data_ann'
        )
        evaluator_replay.print_detailed_table(results_replay)
        evaluator_replay.print_summary(results_replay)
        
    else:
        # Run single evaluation
        if args.ground_truth == 'all':
            ground_truth_dirs = ['replay_data_ann', 'llama-ann']
            suffix = ""
        elif args.ground_truth == 'llama-ann':
            ground_truth_dirs = ['llama-ann']
            suffix = "_llama_ann"
        elif args.ground_truth == 'replay_data_ann':
            ground_truth_dirs = ['replay_data_ann']
            suffix = "_replay_data_ann"
        
        evaluator = AnnotationEvaluator(args.annotation_dir, ground_truth_dirs)
        
        print("Starting annotation evaluation...")
        print(f"Categories: {evaluator.categories}")
        print(f"Ground truth directories: {evaluator.ground_truth_dirs}")
        
        all_results = evaluator.run_evaluation(
            models=args.models, 
            save_results=not args.no_save,
            filename_suffix=suffix
        )
        
        evaluator.print_detailed_table(all_results)
        evaluator.print_summary(all_results)


if __name__ == '__main__':
    main()