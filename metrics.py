"""
Task-specific evaluation metrics for IndicGenBench

This module provides specialized metrics for each task in IndicGenBench:
- CrossSum: ROUGE metrics for summarization
- Flores: BLEU, ChrF, and METEOR for translation
- XQuAD/XorQA: Exact Match and F1 for QA tasks
"""

import re
import numpy as np
import string
from typing import List, Dict, Any, Tuple
from collections import Counter
from sacrebleu import corpus_bleu, corpus_chrf
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

class IndicGenBenchMetrics:
    """Task-specific metrics for IndicGenBench evaluation"""
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer for exact match and F1 calculation in QA tasks"""
        # Remove articles
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        # Remove punctuation
        s = s.translate(str.maketrans('', '', string.punctuation))
        # Lowercase and whitespace normalization
        return ' '.join(s.lower().split())
    
    @staticmethod
    def get_tokens(s: str) -> List[str]:
        """Get tokens for F1 calculation in QA tasks"""
        if not s:
            return []
        return IndicGenBenchMetrics.normalize_answer(s).split()
    
    @staticmethod
    def compute_exact_match(prediction: str, ground_truth: str) -> float:
        """Compute exact match for QA tasks"""
        return float(
            IndicGenBenchMetrics.normalize_answer(prediction) == 
            IndicGenBenchMetrics.normalize_answer(ground_truth)
        )
    
    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        """Compute F1 score for QA tasks"""
        prediction_tokens = IndicGenBenchMetrics.get_tokens(prediction)
        ground_truth_tokens = IndicGenBenchMetrics.get_tokens(ground_truth)
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    @staticmethod
    def compute_qa_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute QA metrics (Exact Match and F1)"""
        exact_match_scores = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # Handle multiple reference answers
            if isinstance(ref, list):
                # Use the best score among all reference answers
                em_scores = [IndicGenBenchMetrics.compute_exact_match(pred, r) for r in ref]
                f1_scores_multi = [IndicGenBenchMetrics.compute_f1(pred, r) for r in ref]
                
                exact_match_scores.append(max(em_scores))
                f1_scores.append(max(f1_scores_multi))
            else:
                exact_match_scores.append(IndicGenBenchMetrics.compute_exact_match(pred, ref))
                f1_scores.append(IndicGenBenchMetrics.compute_f1(pred, ref))
        
        return {
            'exact_match': 100.0 * np.mean(exact_match_scores),
            'f1': 100.0 * np.mean(f1_scores)
        }
    
    @staticmethod
    def compute_bleu(predictions: List[str], references: List[List[str]]) -> float:
        """Compute BLEU score for translation task"""
        return corpus_bleu(predictions, references).score
    
    @staticmethod
    def compute_chrf(predictions: List[str], references: List[List[str]]) -> float:
        """Compute ChrF score for translation task"""
        return corpus_chrf(predictions, references).score
    
    @staticmethod
    def compute_meteor(predictions: List[str], references: List[str]) -> float:
        """Compute METEOR score"""
        scores = []
        for pred, ref in zip(predictions, references):
            # Handle list of references
            if isinstance(ref, list):
                ref = ref[0]
            
            try:
                score = meteor_score([ref.split()], pred.split())
                scores.append(score)
            except Exception:
                # Skip in case of errors
                continue
        
        return 100.0 * np.mean(scores) if scores else 0.0
    
    @staticmethod
    def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores for summarization task"""
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {metric: [] for metric in ['rouge1', 'rouge2', 'rougeL']}
        
        for pred, ref in zip(predictions, references):
            # Handle list of references
            if isinstance(ref, list):
                ref = ref[0]
                
            try:
                results = rouge_scorer_obj.score(ref, pred)
                for metric in scores:
                    scores[metric].append(results[metric].fmeasure)
            except Exception:
                # Skip in case of errors
                continue
        
        return {
            metric: 100.0 * np.mean(vals) if vals else 0.0
            for metric, vals in scores.items()
        }
    
    @staticmethod
    def evaluate_crosssum(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate CrossSum task (summarization)"""
        # Primary metric for summarization is ROUGE
        rouge_scores = IndicGenBenchMetrics.compute_rouge(predictions, references)
        
        # Also compute METEOR for additional perspective
        meteor = IndicGenBenchMetrics.compute_meteor(predictions, references)
        
        return {
            **rouge_scores,
            'meteor': meteor
        }
    
    @staticmethod
    def evaluate_flores(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Evaluate Flores task (translation)"""
        # Primary metrics for translation are BLEU and ChrF
        bleu = IndicGenBenchMetrics.compute_bleu(predictions, references)
        chrf = IndicGenBenchMetrics.compute_chrf(predictions, references)
        
        # Also compute METEOR for additional perspective
        meteor = IndicGenBenchMetrics.compute_meteor(
            predictions, 
            [ref[0] for ref in references]  # Take first reference for METEOR
        )
        
        return {
            'bleu': bleu,
            'chrf': chrf,
            'meteor': meteor
        }
    
    @staticmethod
    def evaluate_xquad(predictions: List[str], references: List[Any]) -> Dict[str, float]:
        """Evaluate XQuAD task (QA)"""
        # Process references to handle different formats
        processed_references = []
        for ref in references:
            if isinstance(ref, list):
                # Use the first answer if reference is a list
                processed_references.append(ref[0] if ref else "")
            elif isinstance(ref, dict) and "text" in ref:
                processed_references.append(ref["text"])
            else:
                processed_references.append(ref)
                
        return IndicGenBenchMetrics.compute_qa_metrics(predictions, processed_references)
    
    @staticmethod
    def evaluate_xorqa(predictions: List[str], references: List[Any]) -> Dict[str, float]:
        """Evaluate XorQA task (cross-lingual QA)"""
        # Process references to handle different formats
        processed_references = []
        for ref in references:
            if isinstance(ref, list):
                # Use the first answer if reference is a list
                processed_references.append(ref[0] if ref else "")
            elif isinstance(ref, dict) and "text" in ref:
                processed_references.append(ref["text"])
            else:
                processed_references.append(ref)
                
        return IndicGenBenchMetrics.compute_qa_metrics(predictions, processed_references)
    
    @staticmethod
    def evaluate(task: str, predictions: List[str], references: List[Any]) -> Dict[str, float]:
        """Evaluate predictions for a specific task"""
        if task == "crosssum_in":
            return IndicGenBenchMetrics.evaluate_crosssum(predictions, references)
        elif task == "flores_in":
            # Convert references to expected format for BLEU if needed
            if references and not isinstance(references[0], list):
                refs_for_bleu = [[ref] for ref in references]
            else:
                refs_for_bleu = references
            return IndicGenBenchMetrics.evaluate_flores(predictions, refs_for_bleu)
        elif task == "xquad_in":
            return IndicGenBenchMetrics.evaluate_xquad(predictions, references)
        elif task == "xorqa_in":
            return IndicGenBenchMetrics.evaluate_xorqa(predictions, references)
        else:
            raise ValueError(f"Unknown task: {task}")