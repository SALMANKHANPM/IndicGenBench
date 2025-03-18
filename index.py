import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Ensure NLTK data is downloaded
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

# Constants
TASKS = ["crosssum_in", "flores_in", "xquad_in", "xorqa_in"]
SPLITS = ["dev", "test"]  # You can also add "train" if needed for analysis

class IndicGenBenchEvaluator:
    def __init__(self, data_dir: str, models: List[str], languages: List[str] = None):
        """
        Initialize the evaluator
        
        Args:
            data_dir: Path to the IndicGenBench data directory
            models: List of model names or paths to evaluate
            languages: List of language codes to evaluate (default: all available)
        """
        self.data_dir = data_dir
        self.models = models
        self.model_pipelines = {}
        
        # Initialize language list if not provided
        if languages is None:
            self.languages = self._discover_languages()
        else:
            self.languages = languages
            
        # Load models
        self._load_models()
    
    def _discover_languages(self) -> List[str]:
        """Discover all available languages in the dataset"""
        languages = set()
        for task in TASKS:
            task_dir = os.path.join(self.data_dir, task)
            if not os.path.exists(task_dir):
                continue
                
            for filename in os.listdir(task_dir):
                if filename.endswith(".json"):
                    parts = filename.replace(".json", "").split("_")
                    if task == "flores_in":
                        # Extract both source and target languages
                        if len(parts) >= 4:  # Make sure we have enough parts
                            languages.add(parts[1])
                            languages.add(parts[2])
                    elif task == "crosssum_in" and "english" in parts[0]:
                        # Format: crosssum_english-{lang}_{split}.json
                        if len(parts) >= 2 and "-" in parts[0]:
                            lang = parts[0].split("-")[1]
                            languages.add(lang)
                    else:
                        # Format: {task}_{lang}_{split}.json
                        if len(parts) >= 2:
                            languages.add(parts[1])
        
        # Remove 'en' (English) if it's in the set since we're focusing on Indic languages
        if 'en' in languages:
            languages.remove('en')
            
        return sorted(list(languages))
    
    def _load_models(self):
        """Load all specified models with Unsloth optimization if available"""
        for model_name in self.models:
            try:
                print(f"Loading model: {model_name}")
                
                # Try to use Unsloth for faster inference
                try:
                    from unsloth import FastLanguageModel
                    
                    print(f"Using Unsloth optimization for: {model_name}")
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_name,
                        max_seq_length=2048,  # Adjust based on your needs
                        load_in_4bit=True,    # Use 4-bit quantization for memory efficiency
                    )
                    # Enable faster inference
                    FastLanguageModel.for_inference(model)
                    
                    # Create generation pipeline
                    from transformers import pipeline
                    self.model_pipelines[model_name] = pipeline(
                        "text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        temperature=0.7
                    )
                    
                    print(f"Successfully loaded model with Unsloth: {model_name}")
                
                except ImportError:
                    # Fall back to standard HuggingFace if Unsloth is not available
                    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                    
                    print(f"Unsloth not available, using standard HuggingFace: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    self.model_pipelines[model_name] = pipeline(
                        "text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        temperature=0.7
                    )
                    
                    print(f"Successfully loaded model: {model_name}")
                    
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                
    def _load_data(self, task: str, language: str, split: str) -> List[Dict[str, Any]]:
        """Load data for a specific task, language and split"""
        filepath = None
        
        if task == "crosssum_in":
            filepath = os.path.join(self.data_dir, task, f"crosssum_english-{language}_{split}.json")
        elif task == "flores_in":
            # Try both en-to-lang and lang-to-en
            filepath_en_to_lang = os.path.join(self.data_dir, task, f"flores_en_{language}_{split}.json")
            filepath_lang_to_en = os.path.join(self.data_dir, task, f"flores_{language}_en_{split}.json")
            
            if os.path.exists(filepath_en_to_lang):
                filepath = filepath_en_to_lang
            elif os.path.exists(filepath_lang_to_en):
                filepath = filepath_lang_to_en
        elif task in ["xquad_in", "xorqa_in"]:
            filepath = os.path.join(self.data_dir, task, f"{task[:-3]}_{language}_{split}.json")
        
        if filepath is None or not os.path.exists(filepath):
            print(f"File not found for {task}-{language}-{split}")
            return []
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            examples = data.get("examples", [])
            
            # Add language code if not already present
            for example in examples:
                if "lang" not in example:
                    example["lang"] = language
                    
                # Add source/target language fields for flores
                if task == "flores_in":
                    if filepath == filepath_en_to_lang:
                        example["src_lang"] = "en"
                        example["tgt_lang"] = language
                    elif filepath == filepath_lang_to_en:
                        example["src_lang"] = language
                        example["tgt_lang"] = "en"
                    
            return examples
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []
    
    def _prepare_prompt(self, task: str, example: Dict[str, Any]) -> str:
        """Prepare the prompt for a specific task and example"""
        if task == "crosssum_in":
            return f"Summarize the following English text in {example.get('lang', '')}:\n\n{example.get('text', '')}"
        
        elif task == "flores_in":
            # Check direction of translation
            src_lang = example.get('src_lang', '')
            tgt_lang = example.get('tgt_lang', '')
            
            if src_lang == 'en':
                return f"Translate the following text from English to {tgt_lang}:\n\n{example.get('source', '')}"
            else:
                return f"Translate the following text from {src_lang} to English:\n\n{example.get('source', '')}"
        
        elif task == "xquad_in":
            return f"Answer the following question based on the given passage in {example.get('lang', '')}:\n\nPassage: {example.get('context', '')}\n\nQuestion: {example.get('question', '')}"
        
        elif task == "xorqa_in":
            return f"Answer the following question in {example.get('lang', '')} based on the English passage:\n\nPassage: {example.get('context', '')}\n\nQuestion: {example.get('question', '')}"
            
        return ""
    
    def _get_reference(self, task: str, example: Dict[str, Any]) -> Any:
        """Get the reference (gold) output for a specific task and example"""
        if task == "crosssum_in":
            return example.get('summary', '')
        
        elif task == "flores_in":
            return example.get('target', '')
        
        elif task == "xquad_in":
            if "answers" in example and isinstance(example["answers"], list):
                # Handle standard xquad format with answers as a list of objects
                answers = [ans["text"] for ans in example["answers"] if "text" in ans]
                return answers[0] if answers else ""
            else:
                # Fallback to generic answer field
                return example.get('answer', '')
            
        elif task == "xorqa_in":
            if "translated_answers" in example and isinstance(example["translated_answers"], list):
                # Handle translated answers for xorqa
                answers = [ans["text"] for ans in example["translated_answers"] if "text" in ans]
                return answers[0] if answers else ""
            elif "answers" in example:
                if isinstance(example["answers"], list):
                    # Handle answers as a list of objects
                    answers = [ans["text"] for ans in example["answers"] if "text" in ans]
                    return answers[0] if answers else ""
                elif isinstance(example["answers"], str):
                    # Handle answers as a string
                    return example["answers"]
            
            # Fallback to generic answer field
            return example.get('answer', '')
            
        return ""
    
    def evaluate_model(self, model_name: str, task: str, language: str, split: str) -> Dict[str, float]:
        """Evaluate a model on a specific task, language and split"""
        data = self._load_data(task, language, split)
        if not data:
            return {}
            
        pipeline = self.model_pipelines.get(model_name)
        if not pipeline:
            return {}
            
        # Metrics
        bleu_references = []
        bleu_candidates = []
        meteor_scores = []
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {metric: [] for metric in ['rouge1', 'rouge2', 'rougeL']}
        
        # QA metrics
        exact_match_scores = []
        f1_scores = []
            
        for example in tqdm(data[:100], desc=f"Evaluating {model_name} on {task}-{language}-{split}"):
            prompt = self._prepare_prompt(task, example)
            reference = self._get_reference(task, example)
            
            if not prompt or not reference:
                continue
                
            # Generate output
            try:
                output = pipeline(prompt, max_new_tokens=512)[0]['generated_text']
                # Strip the prompt from the output
                output = output[len(prompt):].strip()
                
                # Calculate metrics based on task
                if task == "crosssum_in":
                    # ROUGE scores for summarization
                    rouge_result = rouge_scorer_obj.score(reference, output)
                    for metric in rouge_scores:
                        rouge_scores[metric].append(rouge_result[metric].fmeasure)
                    
                    # METEOR for summarization
                    try:
                        meteor = meteor_score([reference.split()], output.split())
                        meteor_scores.append(meteor)
                    except Exception as e:
                        print(f"Error calculating METEOR: {e}")
                
                elif task == "flores_in":
                    # BLEU for translation
                    bleu_references.append([reference])
                    bleu_candidates.append(output)
                    
                    # METEOR for translation
                    try:
                        meteor = meteor_score([reference.split()], output.split())
                        meteor_scores.append(meteor)
                    except Exception as e:
                        print(f"Error calculating METEOR: {e}")
                        
                elif task in ["xquad_in", "xorqa_in"]:
                    # F1 and Exact Match for QA
                    # Normalize answers
                    norm_output = self._normalize_answer(output)
                    norm_reference = self._normalize_answer(reference)
                    
                    # Exact match
                    exact_match_scores.append(float(norm_output == norm_reference))
                    
                    # F1 score
                    pred_tokens = self._get_tokens(norm_output)
                    ref_tokens = self._get_tokens(norm_reference)
                    
                    common = set(pred_tokens) & set(ref_tokens)
                    if not common:
                        f1_scores.append(0.0)
                    else:
                        precision = len(common) / len(pred_tokens) if pred_tokens else 0
                        recall = len(common) / len(ref_tokens) if ref_tokens else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
                        f1_scores.append(f1)
                    
            except Exception as e:
                print(f"Error generating output: {e}")
                
        # Calculate final metrics based on task
        results = {}
        
        if task == "crosssum_in":
            for metric in rouge_scores:
                if rouge_scores[metric]:
                    results[metric] = np.mean(rouge_scores[metric])
            
            if meteor_scores:
                results['meteor'] = np.mean(meteor_scores)
                
        elif task == "flores_in":
            if bleu_candidates:
                try:
                    bleu = corpus_bleu(bleu_candidates, bleu_references)
                    results['bleu'] = bleu.score
                except Exception as e:
                    print(f"Error calculating BLEU: {e}")
                    
            if meteor_scores:
                results['meteor'] = np.mean(meteor_scores)
                
        elif task in ["xquad_in", "xorqa_in"]:
            if exact_match_scores:
                results['exact_match'] = 100.0 * np.mean(exact_match_scores)
                
            if f1_scores:
                results['f1'] = 100.0 * np.mean(f1_scores)
                
        return results
    
    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for QA evaluation"""
        import re
        import string
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
            
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
            
        def white_space_fix(text):
            return ' '.join(text.split())
            
        return white_space_fix(remove_articles(remove_punc(s.lower())))
    
    def _get_tokens(self, s: str) -> List[str]:
        """Get tokens for F1 calculation"""
        if not s:
            return []
        return s.split()
    
    def evaluate(self, output_dir: str = "./results"):
        """Run the full evaluation"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        for model_name in self.models:
            model_results = {}
            
            for task in TASKS:
                task_results = {}
                
                for language in self.languages:
                    language_results = {}
                    
                    for split in SPLITS:
                        results = self.evaluate_model(model_name, task, language, split)
                        if results:
                            language_results[split] = results
                    
                    if language_results:
                        task_results[language] = language_results
                
                if task_results:
                    model_results[task] = task_results
            
            if model_results:
                all_results[model_name] = model_results
                
                # Save individual model results
                with open(os.path.join(output_dir, f"{model_name.replace('/', '_')}_results.json"), 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, indent=2)
        
        # Save combined results
        with open(os.path.join(output_dir, "all_results.json"), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
            
        # Generate comparison report
        self._generate_comparison_report(all_results, output_dir)
        
        return all_results
    
    def _generate_comparison_report(self, all_results: Dict[str, Any], output_dir: str):
        """Generate a comparison report between models"""
        report = []
        report.append("# IndicGenBench Model Comparison Report")
        report.append("")
        
        # Task-wise comparison
        for task in TASKS:
            task_models = {}
            
            # Collect results for all models for this task
            for model_name, model_results in all_results.items():
                if task in model_results:
                    task_models[model_name] = model_results[task]
            
            if not task_models:
                continue
                
            report.append(f"## {task.upper()}")
            report.append("")
            
            # Language-wise comparison
            for language in self.languages:
                # language_models = {}
                
                # Check if any model has results for this language
                has_language = False
                for model_name, task_result in task_models.items():
                    if language in task_result:
                        has_language = True
                        break
                        
                if not has_language:
                    continue
                    
                report.append(f"### {language}")
                report.append("")
                
                # Create table header
                table_header = "| Model | Split | BLEU | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L |"
                table_divider = "| --- | --- | --- | --- | --- | --- | --- |"
                report.append(table_header)
                report.append(table_divider)
                
                # Add rows for each model and split
                for model_name, task_result in task_models.items():
                    if language in task_result:
                        for split, metrics in task_result[language].items():
                            bleu = metrics.get('bleu', 'N/A')
                            meteor = metrics.get('meteor', 'N/A')
                            rouge1 = metrics.get('rouge1', 'N/A')
                            rouge2 = metrics.get('rouge2', 'N/A')
                            rougeL = metrics.get('rougeL', 'N/A')
                            
                            # Format numeric values
                            for metric in [bleu, meteor, rouge1, rouge2, rougeL]:
                                if isinstance(metric, (int, float)):
                                    metric = f"{metric:.2f}"
                            
                            row = f"| {model_name} | {split} | {bleu} | {meteor} | {rouge1} | {rouge2} | {rougeL} |"
                            report.append(row)
                
                report.append("")
        
        # Write the report
        with open(os.path.join(output_dir, "comparison_report.md"), 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on IndicGenBench")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to IndicGenBench data directory")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Model names or paths to evaluate")
    parser.add_argument("--languages", type=str, nargs="+", default=None, help="Language codes to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    
    args = parser.parse_args()
    
    evaluator = IndicGenBenchEvaluator(
        data_dir=args.data_dir,
        models=args.models,
        languages=args.languages
    )
    
    evaluator.evaluate(output_dir=args.output_dir)

if __name__ == "__main__":
    main()