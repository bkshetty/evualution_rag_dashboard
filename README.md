# 🧠 RAG & Baseline Master Evaluation Dashboard

A comprehensive multi-model benchmarking and evaluation platform designed to systematically evaluate Retrieval-Augmented Generation (RAG) architectures against standard baseline models. Powered by **DeepEval** and **LLM-as-a-Judge (70B model)** with automatic rate-limit key rotation.

---

## 🚀 Key Features

*   **Multi-Configuration Matrix**: Evaluates **13 configurations** in total:
    *   **Baseline Model** (LLM Only).
    *   **12 RAG Combinations**: 3 Chunking methods (Fixed, Recursive, Semantic) × 2 Embedding models (SentenceTransformers, BGE-small) × 2 Vector Databases (FAISS, ChromaDB).
*   **Resumable Batch CLI Evaluator (`run_batch_evaluation.py`)**: Runs heavy 200-question evaluation tasks in the background. Tracks progress via `evaluation_progress.json` to allow pause-and-resume capability without duplicate API token costs.
*   **Flexible Range Selection**: Run evaluations on a subset range (e.g. evaluating only questions 50 to 100) or limit configurations.
*   **Rotated 4-Key API Fallback Engine**: Cycles automatically through 4 Groq API keys to bypass rate limits (TPM/RPM throttling) during large runs.
*   **Visual Streamlit Dashboard**: Load historical logs automatically from `evaluation_history_log.csv` and compare architectures using quality metrics, latencies, and radar charts.

---

## 📁 Repository Structure

```text
RAG/
├── README.md                          # Project documentation
├── .gitignore                         # Prevents pushing .env, databases, and venv files
└── rag_evaluation_project/
    ├── 1_generate_dataset.py          # UI script to generate golden dataset from PDFs
    ├── 2_build_databases.py           # Ingestion script to build vector databases
    ├── 3_evaluation_dashboard.py      # Streamlit interactive dashboard UI
    ├── run_batch_evaluation.py        # CLI batch evaluation script (Resumable)
    ├── evaluation_history_log.csv     # Master log file of all metric scores
    ├── evaluation_progress.json       # Checkpoint tracking file for CLI resumption
    ├── data/
    │   ├── ground_truth_dataset.json  # Raw generated Q&A dataset
    │   └── questions.csv              # Current questions list for evaluation
    └── databases/                     # Built Chroma and FAISS indices (ignored by git)
```

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate
Ensure you are in the project folder:
```powershell
cd rag_evaluation_project
```

### 2. Configure Environment Variables
Create a `.env` file inside `rag_evaluation_project/` with your rotated Groq API keys:
```text
GROQ_API_KEY_1=gsk_your_key_1
GROQ_API_KEY_2=gsk_your_key_2
GROQ_API_KEY_3=gsk_your_key_3
GROQ_API_KEY_4=gsk_your_key_4
```

### 3. Build the Vector Databases
Place your study materials/PDF notes in the `data/` folder, then compile the vector databases:
```powershell
.\venv\Scripts\python 2_build_databases.py
```

---

## 🧪 Running Evaluations

### Option A: Run the Resumable CLI Script (Recommended for 200 Qs)
Run the evaluations in your terminal. This is safe from browser disconnects or UI resets:

*   **Evaluate all 200 questions across all 13 configurations:**
    ```powershell
    .\venv\Scripts\python run_batch_evaluation.py --limit-questions 200
    ```

*   **Run only the Baseline configuration for questions 1 to 50:**
    ```powershell
    .\venv\Scripts\python run_batch_evaluation.py --limit-questions 200 --start-question 1 --end-question 50 --limit-configs 1
    ```

*   **Run questions 100 to 150 across all configurations:**
    ```powershell
    .\venv\Scripts\python run_batch_evaluation.py --limit-questions 200 --start-question 100 --end-question 150
    ```

### Option B: Run via Streamlit UI
Start the web dashboard:
```powershell
.\venv\Scripts\streamlit run 3_evaluation_dashboard.py
```
1. Navigate to **Tab 2 (DeepEval Test Automation)**.
2. Select your question range using the range slider.
3. Choose the active RAG configuration in the sidebar.
4. Click **Execute DeepEval Testing Suite**.

---

## 📊 Analytics Dashboard

Launch the dashboard (`streamlit run 3_evaluation_dashboard.py`) and click on **Tab 3 (Master Analytics Dashboard)**:
*   **Quality Metrics Bar Chart**: Compares *Answer Accuracy*, *Retrieval Precision*, and *Hallucination Reduction* (Baseline automatically hides non-applicable retrieval metrics for cleaner comparison).
*   **Latency Comparison**: Plots average response times in seconds for each configuration.
*   **Capability Radar Chart**: Displays the balanced performance profile of each architecture.
