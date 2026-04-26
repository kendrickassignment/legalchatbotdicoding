# Legal Team Chatbot with Fine-Tuned LLM, GRPO, and Advanced RAG

This project is an end-to-end **Generative AI legal assistant** built for an Indonesian Legal Team. It combines **supervised fine-tuning**, **GRPO-based reasoning optimization**, and an **advanced Retrieval-Augmented Generation (RAG)** pipeline to answer legal questions based on official Indonesian regulations.

The system was developed as a final project and covers the full workflow from model adaptation to retrieval and interface deployment.

## Project Overview

This project is divided into three main stages:

1. **Fine-tuning a Small Language Model**
   - Base model: `Qwen2.5-1.5B`
   - Dataset: `Ichsan2895/alpaca-gpt4-indonesian`
   - Method: **QLoRA** with 4-bit loading and LoRA adapters on both attention and feed-forward layers
   - Two supervised fine-tuning experiments were conducted to compare hyperparameter settings

2. **GRPO for Reasoning Behavior**
   - The fine-tuned model was further optimized using **GRPO (Group Relative Policy Optimization)**
   - The objective was to encourage the model to produce explicit reasoning in the format:
     ```text
     <think>...</think>
     ```
   - Four reward functions were designed for format, reasoning length, answer correctness, and Indonesian language consistency

3. **Advanced RAG System**
   - Built using official Indonesian legal PDF documents
   - Includes:
     - metadata enrichment
     - parent-child chunking
     - FAISS vector store
     - BM25 + semantic ensemble retrieval
     - HyDE (Hypothetical Document Embeddings)
     - reranking with a cross-encoder
     - threshold-based fallback to DuckDuckGo
   - Wrapped with a simple **Gradio interface**

## Features

- Fine-tuned Indonesian legal assistant
- GRPO-enhanced reasoning output using `<think>...</think>`
- PDF-based legal knowledge retrieval
- Metadata-aware document handling
- Hybrid retrieval with **BM25 + semantic search**
- Parent-child retrieval for better context reconstruction
- HyDE query expansion
- Cross-encoder reranking
- Internet fallback via DuckDuckGo when local relevance is too low
- Gradio-based chatbot interface

## Tech Stack

- **Python**
- **PyTorch**
- **Unsloth**
- **Transformers**
- **TRL**
- **PEFT / LoRA**
- **LangChain**
- **FAISS**
- **Sentence Transformers**
- **Gradio**

## Model Repositories

- Fine-tuned model:  
  [kendrickfff/Qwen2.5-1.5B-Indonesian-Assistant](https://huggingface.co/kendrickfff/Qwen2.5-1.5B-Indonesian-Assistant)

- GRPO model:  
  [kendrickfff/Qwen2.5-1.5B-Indonesian-Assistant-GRPO](https://huggingface.co/kendrickfff/Qwen2.5-1.5B-Indonesian-Assistant-GRPO)

## Project Structure

```bash
.
├── Fine-tuning_submission_PGABL_Kendrick_Filbert.ipynb
├── GRPO_submission_PGABL_Kendrick_Filbert.ipynb
├── RAG_submission_PGABL_Kendrick_Filbert.ipynb
├── requirements.txt
└── README.md
```

## Fine-Tuning Highlights

- Chat template mapping using Hugging Face `datasets.map()`
- QLoRA with 4-bit quantization
- LoRA adapters applied to:
  - Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - FFN: `gate_proj`, `up_proj`, `down_proj`
- Train/validation split
- Two experiments with different learning rates, schedulers, and effective batch sizes
- Final fine-tuned model uploaded to Hugging Face in **merged 16-bit** format

## GRPO Highlights

The GRPO stage used the fine-tuned model as initialization and introduced four custom reward functions:

- **format_reward_func**  
  Rewards correct `<think>...</think>` structure

- **reasoning_length_reward**  
  Rewards longer and more meaningful reasoning traces

- **correctness_reward**  
  Rewards output similarity to ground-truth answers using ROUGE-L

- **language_reward_func**  
  Rewards Indonesian output and penalizes English answers

## RAG Pipeline Highlights

The RAG system uses four official Indonesian legal documents as the knowledge base and includes:

- explicit chunk size and overlap
- metadata enrichment for document title, type, year, and page
- FAISS for vector search
- BM25 retriever for lexical search
- weighted ensemble retrieval
- parent-child chunk expansion
- HyDE for hypothetical answer-based retrieval
- cross-encoder reranking
- fallback to DuckDuckGo when document relevance is below threshold
- answer generation with citations

## Example Use Case

**Prompt:**  
*“Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?”*

The system retrieves relevant legal passages, reranks them, and generates an answer with source citations.  
In the GRPO version, the model can also produce an explicit reasoning trace using `<think>...</think>`.

## Installation

```bash
pip install -r requirements.txt
```

## Notes

This project was developed with free-tier GPU constraints in mind, so the implementation prioritizes:
- efficient fine-tuning
- memory-aware training settings
- lightweight but capable model selection

## Outcome

This submission was successfully accepted and evaluated positively for:
- comprehensive notebook structure
- complete fine-tuning, GRPO, and RAG workflow
- strong effort in modular function design and testing

## Author

**Kendrick Filbert**

---