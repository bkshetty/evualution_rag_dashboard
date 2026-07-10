import os
import json
import time
import random
import re
import argparse
import sys
import pandas as pd
from dotenv import load_dotenv

# Reconfigure stdout/stderr to UTF-8 to prevent encoding crashes on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# --- AI & LangChain Imports ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from pydantic import BaseModel

# --- DeepEval Testing Framework Imports ---
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, FaithfulnessMetric, ContextualRecallMetric
from deepeval.models.base_model import DeepEvalBaseLLM

# Load environment variables
load_dotenv(override=True)

class GroqFallbackLLM(DeepEvalBaseLLM):
    """DeepEval LLM Wrapper with API key rotation and rate limit backoff."""
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        raw_keys = [
            os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"), os.getenv("GROQ_API_KEY_4")
        ]
        self.api_keys = [k for k in raw_keys if k and k.strip() != ""]
        if not self.api_keys:
            print("WARNING: No Groq API keys found in .env file!")

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema: BaseModel = None):
        """Infinite loop that cycles keys and waits 65s if all keys hit the rate limit."""
        while True:
            for key in self.api_keys:
                try:
                    # max_retries=0 forces it to fail instantly so we can swap keys faster
                    llm = ChatGroq(api_key=key, model_name=self.model_name, temperature=0, max_retries=0)
                    if schema:
                        structured_llm = llm.with_structured_output(schema)
                        result = structured_llm.invoke(prompt)
                        if result is None:
                            continue
                        return result
                    else:
                        return llm.invoke(prompt).content
                except Exception as e:
                    # Swapping keys immediately on exception
                    print(f"      [!] Key {key[:8]}... failed: {e}")
                    continue
            
            print("All API Keys Rate-Limited. Entering 65-second cooldown...")
            time.sleep(65)
            print("Cooldown complete. Resuming...")

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Groq-" + self.model_name


# Direct implementation of Groq JSON generator for question extraction
class GroqJSONGenerator:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        raw_keys = [
            os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"), os.getenv("GROQ_API_KEY_4")
        ]
        self.api_keys = [k for k in raw_keys if k and k.strip() != ""]

    def invoke_json(self, prompt: str):
        while True:
            for key in self.api_keys:
                try:
                    llm = ChatGroq(api_key=key, model_name=self.model_name, temperature=0.2, max_retries=0)
                    messages = [
                        SystemMessage(content="You must output ONLY raw, valid JSON. No markdown wrappers, no intro text."), 
                        HumanMessage(content=prompt)
                    ]
                    response = llm.invoke(messages)
                    clean_text = response.content.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_text)
                except Exception as e:
                    continue
            print("All API Keys Rate-Limited during Q&A generation. Entering 65-second cooldown...")
            time.sleep(65)


def generate_questions_to_target(target_count=200):
    """Generates questions from PDFs to reach target count in data/questions.csv."""
    csv_path = "data/questions.csv"
    existing_questions = []
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Standardize column names
            df.columns = [str(c).strip().lower() for c in df.columns]
            q_col = next((c for c in df.columns if 'q' in c), None)
            a_col = next((c for c in df.columns if 'ground' in c or 'truth' in c or 'answer' in c), None)
            
            if q_col and a_col:
                for idx, row in df.iterrows():
                    existing_questions.append({
                        "Question": row[q_col],
                        "Ground Truth": row[a_col]
                    })
                print(f"[+] Loaded {len(existing_questions)} existing questions from {csv_path}")
        except Exception as e:
            print(f"[-] Error loading existing questions: {e}. Starting fresh.")

    if len(existing_questions) >= target_count:
        print(f"[+] Already have {len(existing_questions)} questions (Target: {target_count}). Skipping generation.")
        return existing_questions[:target_count]

    needed = target_count - len(existing_questions)
    print(f"[+] Generating {needed} more questions to reach target of {target_count}...")
    
    # Load and chunk PDFs
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    if not documents:
        raise Exception("[-] Error: No PDF files found in 'data/' folder.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    
    clean_chunks = []
    for chunk in chunks:
        clean_text = str(chunk.page_content).replace('\x00', '').strip()
        if len(clean_text) > 100:
            chunk.page_content = clean_text
            clean_chunks.append(chunk)
            
    random.seed(42)
    random.shuffle(clean_chunks)
    
    generator = GroqJSONGenerator("llama-3.1-8b-instant")
    consecutive_failures = 0
    
    for idx, chunk in enumerate(clean_chunks):
        if len(existing_questions) >= target_count:
            break
            
        print(f"[*] Extracting Q&A Pair {len(existing_questions)+1}/{target_count}...")
        
        prompt = f"""
        You are an expert technical examiner building an evaluation dataset. 
        Based ONLY on the provided context, generate ONE highly specific, complex question and a perfectly accurate 'Ground Truth' answer.
        
        CRITICAL INSTRUCTIONS FOR GROUND TRUTH:
        1. The Ground Truth must be highly precise, containing at least 2 to 3 sentences.
        2. It must explicitly include the technical keywords, acronyms, or formulas found in the context.
        3. Do not use vague terms like "it" or "they". Be explicit so an automated judge can grade against it.
        
        Context:
        {chunk.page_content}
        
        Output format MUST be strictly JSON:
        {{
          "question": "The specific question here",
          "ground_truth": "The multi-sentence, highly detailed Golden Answer here"
        }}
        """
        
        try:
            qa_pair = generator.invoke_json(prompt)
            if "question" in qa_pair and "ground_truth" in qa_pair:
                existing_questions.append({
                    "Question": qa_pair["question"],
                    "Ground Truth": qa_pair["ground_truth"]
                })
                consecutive_failures = 0
                # Intermediate save
                df_temp = pd.DataFrame(existing_questions)
                os.makedirs("data", exist_ok=True)
                df_temp.to_csv(csv_path, index=False)
            else:
                consecutive_failures += 1
            
            time.sleep(1) # Cooldown
        except Exception as e:
            print(f"[-] Generation error: {e}")
            consecutive_failures += 1
            if consecutive_failures > 10:
                print("[-] Severe API or parsing failures. Stopping generation.")
                break
            time.sleep(5)

    print(f"[+] Golden dataset generation complete. Total questions: {len(existing_questions)}")
    return existing_questions[:target_count]


# Pipelines to retrieve and generate answers
def run_pipeline_without_rag(query: str, generator_llm: GroqFallbackLLM):
    start_time = time.time()
    prompt = f"Answer the following question accurately based solely on your internal knowledge: {query}"
    answer = generator_llm.generate(prompt)
    latency = time.time() - start_time
    return answer, ["N/A"], latency


def run_pipeline_with_rag(query: str, generator_llm: GroqFallbackLLM, chunking: str, embedding: str, db_type: str):
    start_time = time.time()
    
    # 1. Map to correct embedding model (fixes dimension mismatch from large to small)
    embed_model_name = "BAAI/bge-small-en-v1.5" if embedding == "BGE embeddings" else "all-MiniLM-L6-v2"
    embedder = HuggingFaceEmbeddings(model_name=embed_model_name)
    
    c_str = chunking.lower()
    e_str = "bge" if embedding == "BGE embeddings" else "sentence-transformers"
    d_str = "chroma" if db_type == "ChromaDB" else "faiss"
    db_path = f"databases/{c_str}_{e_str}_{d_str}"
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database folder '{db_path}' not found. Please run ingestion first.")
        
    if d_str == "chroma":
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedder)
    else:
        vectorstore = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
        
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.75}
    )
    
    docs = retriever.invoke(query)
    contexts = [doc.page_content for doc in docs]
    
    context_str = "\n\n".join(contexts)
    prompt = f"""You are a highly precise, academic AI evaluator. You are provided with retrieved context from a specialized database.
    
    YOUR DIRECTIVES:
    1. Evaluate the Context: Does it explicitly contain the factual answer to the Question?
    2. If YES: Answer the question using ONLY the facts provided in the Context.
    3. If NO: You must completely IGNORE the Context. Answer the question using your internal expert knowledge.
    4. Do not mention that you are ignoring the context. Just provide the correct answer seamlessly.
    
    [RETRIEVED CONTEXT]
    {context_str}
    
    [QUESTION] 
    {query}
    
    [FINAL ANSWER]:"""
    
    answer = generator_llm.generate(prompt)
    latency = time.time() - start_time
    return answer, contexts, latency


def main():
    parser = argparse.ArgumentParser(description="Automated 200-Question Multi-Model RAG Evaluator")
    parser.add_argument("--limit-questions", type=int, default=200, help="Number of questions to evaluate (default: 200)")
    parser.add_argument("--start-question", type=int, default=1, help="1-indexed index of question to start at (default: 1)")
    parser.add_argument("--end-question", type=int, default=None, help="1-indexed index of question to end at (inclusive, default: all)")
    parser.add_argument("--limit-configs", type=int, default=13, help="Max configurations to run (default: 13, runs all)")
    parser.add_argument("--force-regenerate", action="store_true", help="Force question generation from PDFs even if CSV exists")
    args = parser.parse_args()

    print("=================================================================")
    print("[RUN] Automated RAG & Baseline Multi-Model Evaluation Script")
    print("=================================================================")

    # 1. Generate/Verify questions
    if args.force_regenerate and os.path.exists("data/questions.csv"):
        try:
            os.remove("data/questions.csv")
            print("[+] Removed existing data/questions.csv for regeneration.")
        except Exception as e:
            print(f"[-] Could not remove questions.csv: {e}")
            
    all_questions = generate_questions_to_target(args.limit_questions)
    
    # Store original 1-based index and map for slicing
    questions_with_index = []
    for idx, q_pair in enumerate(all_questions):
        questions_with_index.append({
            "Question": q_pair["Question"],
            "Ground Truth": q_pair["Ground Truth"],
            "orig_idx": idx # 0-based original index
        })
        
    start_idx = max(0, args.start_question - 1)
    end_idx = args.end_question if args.end_question is not None else len(questions_with_index)
    end_idx = min(len(questions_with_index), end_idx)
    
    sliced_questions = questions_with_index[start_idx:end_idx]
    print(f"[+] Sliced question dataset to range: questions {args.start_question} to {end_idx} (Total: {len(sliced_questions)} questions)")
    
    # 2. Setup all 13 configurations
    # Configurations matrix
    chunking_options = ["Fixed", "Recursive", "Semantic"]
    embedding_options = ["SentenceTransformers", "BGE embeddings"]
    db_options = ["FAISS", "ChromaDB"]

    configs = []
    # Configuration 1: Baseline
    configs.append({
        "System Type": "Without RAG (Baseline)",
        "Architecture": "None",
        "Chunking": "None",
        "Embedding": "None",
        "Database": "None"
    })

    # Configurations 2-13: RAG Combinations
    for chunk in chunking_options:
        for embed in embedding_options:
            for db in db_options:
                configs.append({
                    "System Type": "With RAG",
                    "Architecture": "Naive RAG",
                    "Chunking": chunk,
                    "Embedding": embed,
                    "Database": db
                })

    # Apply configuration limit if specified for testing
    configs = configs[:args.limit_configs]
    print(f"[+] Configured to run {len(configs)} configurations on {len(sliced_questions)} questions.")

    # 3. Load checkpoints
    checkpoint_path = "evaluation_progress.json"
    progress = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                progress = json.load(f)
            print(f"[+] Loaded evaluation progress from {checkpoint_path}")
        except Exception as e:
            print(f"[-] Error loading checkpoint: {e}")

    # Initialize LLMs (Uses 8B model to handle massive TPD limits for 1000+ questions)
    evaluator_llm_judge = GroqFallbackLLM("llama-3.1-8b-instant")
    generator_llm = GroqFallbackLLM("llama-3.1-8b-instant")

    # 4. Main Evaluation Loop
    log_file_path = "evaluation_history_log.csv"
    
    for c_idx, config in enumerate(configs):
        config_name = f"{config['System Type']}|{config['Chunking']}|{config['Embedding']}|{config['Database']}"
        print(f"\n=======================================================")
        print(f"[RUN] [{c_idx+1}/{len(configs)}] RUNNING CONFIGURATION: {config_name}")
        print(f"=======================================================")

        if config_name not in progress:
            progress[config_name] = []

        for q_pair in sliced_questions:
            orig_idx = q_pair["orig_idx"]
            if str(orig_idx) in progress[config_name]:
                print(f"   [->] Question {orig_idx+1}/{len(all_questions)} already evaluated. Skipping.")
                continue

            question = q_pair["Question"]
            ground_truth = q_pair["Ground Truth"]
            print(f"\n[*] Evaluating Q{orig_idx+1}/{len(all_questions)}:")
            print(f"    Q: {question[:80]}...")

            # Run pipeline
            try:
                if config["System Type"] == "Without RAG (Baseline)":
                    answer, contexts, latency = run_pipeline_without_rag(question, generator_llm)
                else:
                    answer, contexts, latency = run_pipeline_with_rag(
                        question, generator_llm, 
                        config["Chunking"], config["Embedding"], config["Database"]
                    )
            except Exception as pipe_err:
                print(f"    [-] Pipeline Execution Error: {pipe_err}")
                continue

            # Run DeepEval metric grading
            print("    [JUDGE] Running DeepEval metrics evaluation...")
            test_case = LLMTestCase(
                input=question, 
                actual_output=answer, 
                expected_output=ground_truth, 
                retrieval_context=contexts
            )
            
            try:
                # Answer Accuracy (Relevancy to the question)
                acc_metric = AnswerRelevancyMetric(threshold=0.5, model=evaluator_llm_judge, include_reason=False)
                acc_metric.measure(test_case)
                acc_score = acc_metric.score
                
                if config["System Type"] != "Without RAG (Baseline)":
                    # Contextual Precision (Did it rank the right context highly?)
                    prec_metric = ContextualPrecisionMetric(threshold=0.5, model=evaluator_llm_judge, include_reason=False)
                    prec_metric.measure(test_case)
                    prec_score = prec_metric.score
                    
                    # Hallucination Reduction (Faithfulness to the retrieved context)
                    faith_metric = FaithfulnessMetric(threshold=0.5, model=evaluator_llm_judge, include_reason=False)
                    faith_metric.measure(test_case)
                    hall_score = faith_metric.score
                    
                    # Contextual Recall (Did the context contain the ground truth?)
                    recall_metric = ContextualRecallMetric(threshold=0.5, model=evaluator_llm_judge, include_reason=False)
                    recall_metric.measure(test_case)
                    # We average Recall and Relevancy for a much stricter "True Accuracy" score
                    acc_score = (acc_score + recall_metric.score) / 2
                else:
                    prec_score, hall_score = 0.0, 0.0
            except Exception as eval_err:
                print(f"    [-] Metric grading failed: {eval_err}")
                acc_score, prec_score, hall_score = 0.0, 0.0, 0.0

            # Scale metrics from 0.0-1.0 to 1.0-5.0
            scaled_acc = round(max(1.0, acc_score * 5), 1)
            scaled_prec = round(max(1.0, prec_score * 5), 1)
            scaled_hall = round(max(1.0, hall_score * 5), 1)

            # Compile single result
            single_result = {
                "System Type": config["System Type"],
                "Architecture": config["Architecture"],
                "Chunking": config["Chunking"],
                "Embedding": config["Embedding"],
                "Database": config["Database"],
                "Answer Accuracy": scaled_acc,
                "Retrieval Precision": scaled_prec,
                "Hallucination Reduction": scaled_hall,
                "Response Time (s)": round(latency, 2),
            }

            # Save immediately to CSV
            single_df = pd.DataFrame([single_result])
            if os.path.exists(log_file_path):
                # Check if file has header, if empty write with header
                if os.path.getsize(log_file_path) > 0:
                    single_df.to_csv(log_file_path, mode='a', header=False, index=False)
                else:
                    single_df.to_csv(log_file_path, index=False)
            else:
                single_df.to_csv(log_file_path, index=False)

            # Save progress checkpoint
            progress[config_name].append(str(orig_idx))
            with open(checkpoint_path, "w") as f:
                json.dump(progress, f, indent=4)

            print(f"    [OK] Completed Q{orig_idx+1}: Acc={scaled_acc} | Prec={scaled_prec} | HallucinationRed={scaled_hall} | Latency={latency:.2f}s")
            time.sleep(1) # Basic cooldown between questions

    print("\n=======================================================")
    print("[OK] All Automated evaluations successfully complete and logged!")
    print("=======================================================")


if __name__ == "__main__":
    main()
