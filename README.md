# IndicGenBench: Benchmarking LLMs for Indic Languages

IndicGenBench is a comprehensive benchmark suite for evaluating LLMs across a variety of tasks in Indian languages, including summarization, translation, question answering, and more.

## Features

- Supports evaluation across 9+ Indic languages
- Includes 4 diverse NLP tasks: summarization, translation, QA, and cross-lingual QA
- Compatible with local models, HuggingFace models, OpenAI APIs, and Anthropic APIs
- Provides detailed metrics and visualizations for model performance
- Optimized with Unsloth for faster inference on limited hardware

## Supported Languages

- Assamese (as)
- Bengali (bn)
- Hindi (hi)
- Kannada (kn)
- Malayalam (ml)
- Marathi (mr)
- Tamil (ta)
- Telugu (te)
- Urdu (ur)
- And more...

## Tasks

- **CrossSum**: Cross-lingual summarization (English to Indic language)
- **Flores**: Machine translation (bidirectional between English and Indic languages)
- **XQuAD**: Question answering in Indic languages
- **XorQA**: Cross-lingual QA (Indic question, English context)

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/SukrutAI/IndicGenBench.git
cd IndicGenBench

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Unsloth for faster inference
pip install unsloth
```

### Google Colab Installation

```python
# Mount Google Drive (optional, for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/SukrutAI/IndicGenBench.git
%cd IndicGenBench

# Install dependencies
!pip install -r requirements.txt

# Install Unsloth
!pip install unsloth

# Set up data directory
DATA_DIR = "/content/IndicGenBench_data"  # Change as needed
!mkdir -p $DATA_DIR
```

### RunPod Installation

```bash
# Clone repository
git clone https://github.com/SukrutAI/IndicGenBench.git
cd IndicGenBench

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (should already have PyTorch with CUDA)
pip install unsloth
```

## Dataset Preparation

IndicGenBench requires specific dataset formats for each task. Download the datasets:

```bash
# Create data directory
mkdir -p data

# Download datasets (you may need to adapt this)
# Example:
# wget -O data/crosssum_in.zip https://example.com/crosssum_in.zip
# unzip data/crosssum_in.zip -d data/

# Verify data structure
ls -la data/
```

## Usage

### Configuration

Create a `config.json` file to specify your evaluation parameters:

```json
{
  "models": [
    {
      "name": "unsloth/Llama-3.2-1B",
      "type": "huggingface",
      "device": "cuda"
    }
  ],
  "languages": ["as", "bn", "hi", "kn", "ml", "mr", "ta", "te", "ur"],
  "tasks": ["crosssum_in", "flores_in", "xquad_in", "xorqa_in"],
  "splits": ["dev"],
  "sample_size": 10
}
```

### Running the Benchmark

#### Local or RunPod

```bash
python runner.py --data_dir /path/to/data --config config.json --output_dir ./results
```

#### Google Colab

```python
# Set up paths
DATA_DIR = "/content/data"
OUTPUT_DIR = "/content/drive/MyDrive/IndicGenBench_results"  # If using Drive

# Create a config file
%%writefile config.json
{
  "models": [
    {
      "name": "unsloth/Llama-3.2-1B",
      "type": "huggingface",
      "device": "cuda"
    }
  ],
  "languages": ["hi", "bn", "ta"],
  "tasks": ["crosssum_in", "flores_in"],
  "splits": ["dev"],
  "sample_size": 5
}

# Run the benchmark
!python runner.py --data_dir $DATA_DIR --config config.json --output_dir $OUTPUT_DIR --num_workers 1
```

### Using with OpenAI or Anthropic Models

To use API-based models, update your config:

```json
{
  "models": [
    {
      "name": "gpt-4o",
      "type": "openai",
      "api_key": "YOUR_API_KEY"  // or use environment variable
    },
    {
      "name": "claude-3.5-haiku",
      "type": "anthropic",
      "api_key": "YOUR_API_KEY"  // or use environment variable
    }
  ],
  "languages": ["hi", "ta"],
  "tasks": ["crosssum_in"],
  "splits": ["dev"],
  "sample_size": 5
}
```

For API keys, you can also use environment variables:

```bash
# On local or RunPod
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# On Colab
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
```

## Available Metrics

The benchmark provides different metrics based on the task:

- **CrossSum**: ROUGE-1, ROUGE-2, ROUGE-L, METEOR
- **Flores**: BLEU, ChrF, METEOR
- **XQuAD/XorQA**: Exact Match, F1 Score

## Example Colab Notebook

Here's a complete example for setting up and running IndicGenBench in Colab:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install transformers torch unsloth nltk sacrebleu rouge-score pandas matplotlib seaborn

# Clone repository (replace with your actual repo)
!git clone https://github.com/SukrutAI/IndicGenBench.git
%cd IndicGenBench

# Download sample data (you'll need to adapt this for your actual data)
!mkdir -p data
# Download your data here

# Create configuration
%%writefile config.json
{
  "models": [
    {
      "name": "unsloth/Llama-3.2-1B",
      "type": "huggingface",
      "device": "cuda"
    }
  ],
  "languages": ["hi", "bn"],
  "tasks": ["crosssum_in"],
  "splits": ["dev"],
  "sample_size": 3
}

# Run evaluation
!python runner.py --data_dir ./data --config config.json --output_dir ./results

# View results
!ls -la ./results
!cat ./results/summary_report.md
```

## RunPod Example

For RunPod, which provides GPU instances, you can use a startup script:

```bash

# Clone repository
git clone https://github.com/SukrutAI/IndicGenBench.git
cd IndicGenBench

# Install dependencies
pip install -r requirements.txt
pip install unsloth

# Download data (adapt as needed)
mkdir -p data
# Download your data here

# Create config
cat > config.json << EOL
{
  "models": [
    {
      "name": "unsloth/Llama-3.2-1B",
      "type": "huggingface", 
      "device": "cuda"
    }
  ],
  "languages": ["hi", "bn", "ta", "te"],
  "tasks": ["crosssum_in", "flores_in"],
  "splits": ["dev"],
  "sample_size": 10
}
EOL

# Run benchmark
python runner.py --data_dir ./data --config config.json --output_dir ./results
```

When setting up your RunPod instance:
1. Upload this script or include it in your template
2. Set it as your startup script
3. Choose an appropriate GPU (A100, H100, etc.)
4. Start the instance

## Interpreting Results

After running the benchmark, check the `results` directory for:

- `summary_report.md`: Overall performance summary
- `*_detailed_report.md`: Detailed per-model reports
- `results.csv`: Raw results in CSV format
- `visualizations/`: Performance visualization charts

## License

[MIT License](LICENSE)

## Citation

If you use this benchmark in your research, please cite:

```
@software{IndicGenBench,
  author = {Himanshu Maurya},
  title = {IndicGenBench: A Benchmark for Generative Language Models for Indic Languages},
  year = {2025},
  url = {https://github.com/SukrutAI/IndicGenBench}
}
```