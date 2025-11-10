# 📚 RAG-Based Question Answering System using LangChain & Hugging Face

This project implements a **Retrieval-Augmented Generation (RAG)** based **Question Answering (Q&A)** system using **LangChain**, **Hugging Face LLMs**, and **ChromaDB** as the vector database.

The goal of this project is to **build, compare, and evaluate** multiple retrieval and text chunking strategies to determine which combination gives the most contextually accurate and efficient answers for document-based Q&A.

---

## 🧠 Project Overview

This notebook performs **document ingestion, chunking, vector embedding, retrieval, and question answering** using LLMs.  
It allows querying large PDFs such as research papers, books, or reports through natural language.

---

## ⚙️ Workflow

1. **Load Documents**
   - Load multiple PDF files using:
     - `PyMuPDFLoader` (for individual files)
     - `DirectoryLoader` (for batch loading from the `Books/` folder)

2. **Split Documents**
   - Text is divided into smaller chunks to improve embedding and retrieval efficiency.

3. **Embed Text**
   - Each chunk is converted into numerical vectors using **Hugging Face embeddings** (`google/embeddinggemma-300m`).

4. **Store Vectors**
   - Embeddings are stored in a **ChromaDB vector store** for efficient retrieval.

5. **Retrieve Context**
   - When a user asks a question, the retriever finds the most relevant text chunks.

6. **Generate Answer**
   - The retrieved context is passed to a **Hugging Face LLM** (`openai/gpt-oss-120b`) to generate the final natural language answer.

---

## 🧩 Methods Compared

### 🧱 Text Splitting Techniques

| Splitter Type | Description | Purpose | Result |
|----------------|--------------|----------|---------|
| **RecursiveCharacterTextSplitter** | Splits text by paragraphs, sentences, or characters recursively with fixed chunk size and overlap | Ensures context continuity | ✅ **Best for small & medium documents** (fast and consistent) |
| **SemanticChunker** | Uses **semantic similarity** to split text dynamically based on meaning rather than length | Improves retrieval precision and contextual grouping | 🧠 **Best for complex documents**, but slower and version-dependent |

---

### 🔍 Retriever Types

| Retriever Type | Description | Purpose | Performance |
|------------------|-------------|----------|--------------|
| **Similarity Retriever** | Standard similarity-based retriever using cosine similarity | Baseline model | ⚙️ Simple and consistent |
| **MultiQueryRetriever** | Expands the user query into multiple semantically equivalent sub-queries | Improves recall and diversity of retrieved documents | 🧠 Best overall performance in complex Q&A |
| **ContextualCompressionRetriever** | Compresses retrieved chunks to retain only the most relevant sentences | Reduces noise and improves precision | ✅ Good balance of precision and recall |
| **Maximum Marginal Retervier** | Combines Similarity retrievers (e.g., basic + Deleting the Duplicate Documents) with a weighted scoring mechanism | Leverages strengths of multiple methods |

---

## 🧪 Experimental Setup

- **Embedding Model:** `google/embeddinggemma-300m`
- **LLM Model:** `openai/gpt-oss-120b(120 Billion Parameters Trained LLM)`
- **Vector Store:** ChromaDB (Persistent local storage)
- **Dataset:** Custom uploaded PDFs (via the `Books/` folder)

---

## 🧩 Findings & Best Combination

After testing all retrieval and chunking combinations, the following setup produced the **most contextually accurate answers**:

| Component | Best Performer | Reason |
|------------|----------------|--------|
| **Text Splitter** | `RecursiveCharacterTextSpiltter` | Groups Words and Paragraphs similar text better, reducing context fragmentation |
| **Retriever** | ` ContextualCompressorRetervier` | Expands user intent, retrieving more relevant passages and help to delete the ambigous Text|

✅ **Final Recommended Pipeline:**
```

RecursiveCharacterTextSpiltter  +  ContextualCompressor  +  google/embeddinggemma-300m  +  openai/gpt-oss-120b

```

---

## ⚠️ SemanticChunker Version Note

The `SemanticChunker` module was part of **`langchain_experimental.text_splitter`**.  
However, **LangChain frequently restructures its experimental modules**, and this import may no longer work in newer versions.

### If you encounter:
```

ModuleNotFoundError: No module named 'langchain_experimental.text_splitter'

````

### Fix Options ( Once it worked Please go through the Langchain Documentation and Please Verify due to PyPI will be regularly updates everyday in langchain experimental so please Go through the Documentation and verify or with 

from langchain_experimental.text_splitter import Semantic Chunker then Verify 

#### ✅ Option 1 — Use the older stable version
```bash
pip install langchain-experimental==0.0.57
````

#### ✅ Option 2 — Use the new import path

```python
from langchain_community.text_splitter import SemanticChunker
```

#### ✅ Option 3 — Fallback to stable text splitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

> 💡 *Tip:* The standard `RecursiveCharacterTextSplitter` is recommended for production or deployment due to stability.

---

## 🛠️ Installation

1. Clone this repository:

   ```bash
   git clone https://Siva-surya-yadala-venkata/RAG-Based-Q-and-A.git
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   It's in the ipynb file
   ```

3. Place your PDF files inside the `Books/` folder.

4. Open the notebook:

   ```bash
   jupyter notebook "Rag Based Q and A.ipynb"
   ```

5. Enter your Hugging Face API token and run all cells.

---

## 📦 Example `requirements.txt`

```text
langchain
langchain-core
langchain-community
langchain-experimental==0.0.57
chromadb
pymupdf
huggingface-hub
```

---

## 🧠 Future Work

* Add Streamlit-based web interface (`app.py`)
* Add evaluation metrics (context recall, precision, latency) where langchain.evaluation is on working mode 
* Support for hybrid retrievers (vector + keyword search)
* Integrate caching for faster repeated queries

---

## ✍️ Author

**Yadala Venkata Siva Surya**
📬 Contact: [[LinkedIn ](https://www.linkedin.com/in/yadala-venkata-siva-surya-1a1a3b256/)/ [GitHub profile link here](https://github.com/Siva-surya-yadala-venkata/)]
🕉️ *Om Nama Sivaya*
