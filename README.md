# Cache-Augmented Generation (CAG)

A practical implementation of prompt caching using Llama 3.2-3B to efficiently answer multiple questions about a document (resume) without reprocessing the source material.

## 🎯 What is Cache-Augmented Generation?

Cache-Augmented Generation leverages transformer KV (key-value) caches to avoid redundant computation. Instead of tokenizing and processing the same context for every query, CAG:

1. **Preprocesses context once** — The source document is tokenized and a forward pass generates cached key-value states
2. **Reuses cached states** — Subsequent queries skip reprocessing the context; only new question tokens are computed
3. **Cleans cache between queries** — The KV cache is restored to its original state, removing query-specific activations

This pattern dramatically speeds up multi-turn reasoning and Q&A over fixed contexts.

## 📊 Performance Benefit

- **First query**: Pays full cost (context preprocessing + answer generation)
- **Subsequent queries**: Only compute answer tokens—context already cached
- **Use case**: Perfect for chatbots, RAG systems, and document analyzers where the same context is queried multiple times

## 🏗️ Architecture

### Core Components

**`preprocess_knowledge()`**
- Tokenizes the knowledge source (resume)
- Runs a forward pass to populate the KV cache
- Returns a `DynamicCache` object containing cached key-value pairs

**`prepare_kvcache()`**
- Wraps knowledge in system instructions
- Formats the prompt for the Llama chat template
- Calls `preprocess_knowledge()` to generate the cache
- Stores original cache length for later restoration

**`clean_up()`**
- Crops the KV cache back to its original length
- Removes query and answer tokens from the cache
- Resets the cache state for the next query

**`generate()`**
- Custom greedy decoder
- Reuses the pre-built KV cache
- Generates answer tokens one at a time
- Stops at EOS (`<|eot_id|>`) or end-of-sequence tokens

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers langchain_community pypdf
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# Load model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

# Load your document
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()
knowledge = "\n\n".join(doc.page_content for doc in documents)

# Create knowledge cache (one-time cost)
knowledge_cache, kv_len = prepare_kvcache(
    documents=knowledge,
    answer_instruction="Answer questions using only the provided document."
)

# Query 1
query1 = "What is X?"
clean_up(knowledge_cache, kv_len)
answer1 = generate_answer(query1, knowledge_cache)

# Query 2 (reuses cache)
query2 = "Tell me about Y"
clean_up(knowledge_cache, kv_len)
answer2 = generate_answer(query2, knowledge_cache)
```

## 📝 Example Queries

The notebook demonstrates three example queries on a resume:

1. **"What experience does the candidate have with Large Language Models?"** — Extracts specific technical experience
2. **"What projects has the candidate built?"** — Lists relevant project history
3. **"What is the github url link?"** — Retrieves contact/profile information

All three queries reuse the same cached resume context.

## 🔧 Technical Details

### KV Cache Lifecycle

1. **Initialize**: `DynamicCache()` creates an empty cache
2. **Populate**: Forward pass with `use_cache=True` fills the cache
3. **Reuse**: Subsequent forward passes append to the existing cache
4. **Crop**: `cache.crop(original_length)` removes query/answer tokens
5. **Repeat**: Ready for the next query

### Why It Works

Transformers compute attention over the full sequence. The KV cache stores pre-computed keys and values for all context tokens, so:
- No need to recompute attention on context for each query
- Only new query tokens are processed through full attention
- Cache is stateful but restorable via `crop()`

### Model & Tokenizer

- **Model**: Meta Llama 3.2-3B-Instruct (3 billion parameters, instruction-tuned)
- **Tokenizer**: Llama chat template with special tokens like `<|start_header_id|>` and `<|eot_id|>`
- **Decoding**: Greedy selection (argmax over logits)
- **Device**: Auto-mapped (GPU preferred if available)

## 📦 Files

- **`CAG.ipynb`** — Full implementation with example queries
- **`README.md`** — This file

## 💡 Extensions & Improvements

- **Batch queries**: Process multiple queries in parallel using the shared cache
- **Sliding window**: Implement cache rotation for very long documents
- **Beam search**: Replace greedy decoding with beam search for better answers
- **Multi-document**: Preprocess multiple documents and chain them in the cache
- **Retrieval-augmented generation (RAG)**: Combine with vector search to select relevant chunks before caching
- **Quantization**: Use 4-bit or 8-bit quantization to reduce memory footprint (commented code in notebook)

## ⚙️ Configuration

Edit these variables in the notebook:

```python
model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Model choice
max_new_tokens = 700  # Maximum answer length
pdf_path = "path/to/resume.pdf"  # Your document
answer_instruction = "..."  # System instructions
```

## 🔐 Authentication

The notebook uses `notebook_login()` to authenticate with Hugging Face Hub for model access. Ensure you have a valid HF token before running.

## 📚 References

- [Transformers KV Cache Documentation](https://huggingface.co/docs/transformers/kv_cache)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [LangChain PDF Loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)

## License

Open source—feel free to use and adapt for your projects.
