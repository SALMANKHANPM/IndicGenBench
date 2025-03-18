"""
This file contains prompt templates for each task in IndicGenBench.
These templates are designed to be consistent across different models.
"""

TASK_DESCRIPTIONS = {
    "crosssum_in": "Given an English article, generate a concise summary in {language}.",
    "flores_in": "Translate the text from {source_language} to {target_language} accurately.",
    "xquad_in": "Answer the question in {language} based on the provided passage in {language}.",
    "xorqa_in": "Answer the question in {language} based on the provided English passage."
}

# CrossSum: English-to-Indic summarization
CROSSSUM_PROMPT = """# Task: Cross-lingual Summarization
# Description: Summarize the following English text in {language}.
# Instructions: Provide a concise and accurate summary of the main points in {language}.

English Text:
{text}

Summary in {language}:
"""

# Flores: Translation in both directions
FLORES_EN_TO_INDIC_PROMPT = """# Task: Machine Translation
# Description: Translate from English to {language}.
# Instructions: Provide an accurate and fluent translation.

English Text:
{source}

{language} Translation:
"""

FLORES_INDIC_TO_EN_PROMPT = """# Task: Machine Translation
# Description: Translate from {language} to English.
# Instructions: Provide an accurate and fluent translation.

{language} Text:
{source}

English Translation:
"""

# XQuAD: Indic QA (question and context in same language)
XQUAD_PROMPT = """# Task: Question Answering
# Description: Answer the question in {language} based on the passage in {language}.
# Instructions: Extract the answer from the passage. Your answer should be short and precise.

Passage ({language}):
{context}

Question ({language}):
{question}

Answer ({language}):
"""

# XorQA: Cross-lingual QA (Indic question, English context)
XORQA_PROMPT = """# Task: Cross-lingual Question Answering
# Description: Answer the {language} question based on the English passage.
# Instructions: Extract the answer from the passage and provide it in {language}.

Passage (English):
{context}

Question ({language}):
{question}

Answer ({language}):
"""

def get_prompt_for_task(task, example, language_name=None):
    """Generate the appropriate prompt based on task and example"""
    if language_name is None:
        language_name = example.get('lang', '')
    
    if task == "crosssum_in":
        return CROSSSUM_PROMPT.format(
            language=language_name,
            text=example.get('text', '')
        )
    
    elif task == "flores_in":
        # Check direction (en->indic or indic->en)
        src_lang = example.get('src_lang', '')
        # tgt_lang = example.get('tgt_lang', '')
        
        if src_lang == 'en':
            return FLORES_EN_TO_INDIC_PROMPT.format(
                language=language_name,
                source=example.get('source', '')
            )
        else:
            return FLORES_INDIC_TO_EN_PROMPT.format(
                language=language_name,
                source=example.get('source', '')
            )
    
    elif task == "xquad_in":
        return XQUAD_PROMPT.format(
            language=language_name,
            context=example.get('context', ''),
            question=example.get('question', '')
        )
    
    elif task == "xorqa_in":
        return XORQA_PROMPT.format(
            language=language_name,
            context=example.get('context', ''),
            question=example.get('question', '')
        )
    
    return ""