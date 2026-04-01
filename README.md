# Hinglish-HUS

Task-aware hallucination analysis for Hinglish (Hindi-English code-switched) prompts, combining the HUS framework with syntactic variation testing.

## What this is
- 180 Hinglish prompts across factual, creative, and brainstorm tasks
- Three prompt variants per item: base, topic_fronted, emphasis_shift
- Generation, annotation, and training pipeline for hallucination labels (N, H-, H+)

## Project layout
- data/  Inputs and outputs
- src/   Pipeline scripts and schemas

## Setup
1. Create a Python environment (3.10+ recommended).
2. Install dependencies:
   - python-dotenv
   - requests
   - tqdm
   - pydantic
   - numpy
   - scikit-learn
   - torch
   - sentence-transformers
3. Add your NIM API key in a .env file:
   - NVIDIA_API_KEY=your_key_here
   - Optional: NIM_BASE_URL=https://integrate.api.nvidia.com/v1

## Run the pipeline
1. Ensure prompts are in data/prompts.json
2. Generate model outputs:
   - python -m src.generate
   - Writes data/raw_outputs_PRESSURED.jsonl
3. Annotate with the judge model:
   - python -m src.annotate
   - Writes data/labeled_PRESSURED.jsonl
4. Train the classifier:
   - python -m src.train
   - Writes data/hinglish_hus_mlp.pt

## Notes
- PRESSURED mode only: models respond confidently without uncertainty.
- Judge prompt treats Hinglish syntactic reordering as non-hallucinatory.
- Metric analysis (BLEU, BERTScore, COMET) is not implemented in src.

## Models
- Generators: meta/llama-3.1-8b-instruct, mistralai/mistral-7b-instruct-v0.2, qwen/qwen2.5-7b-instruct
- Judge: meta/llama-3.1-70b-instruct (temperature 0.0)
