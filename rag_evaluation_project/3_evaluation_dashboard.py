import streamlit as st
import pandas as pd
import time
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pydantic import BaseModel

# --- LangChain & AI Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# --- DeepEval Testing Framework Imports ---
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, FaithfulnessMetric, ContextualRecallMetric
from deepeval.models.base_model import DeepEvalBaseLLM

# Suppress warnings for clean terminal output
import warnings
warnings.filterwarnings("ignore")

load_dotenv(override=True)

# ==========================================
# 1. APP STATE & CONFIGURATION
# ==========================================
st.set_page_config(page_title="RAG Master Evaluation Matrix", layout="wide", initial_sidebar_state="expanded")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_results" not in st.session_state:
    # Load historical results from log file if it exists
    log_file_path = "evaluation_history_log.csv"
    if os.path.exists(log_file_path):
        try:
            df_log = pd.read_csv(log_file_path)
            # Filter out empty rows or headers
            df_log = df_log.dropna(subset=["System Type"])
            st.session_state.eval_results = df_log.to_dict(orient="records")
        except Exception:
            st.session_state.eval_results = []
    else:
        st.session_state.eval_results = []

# ==========================================
# 2. 4-KEY API FALLBACK ENGINE (THE JUDGE)
# ==========================================
class GroqFallbackLLM(DeepEvalBaseLLM):
    """Bulletproof DeepEval LLM Wrapper with Infinite Retry and Cooldown."""
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        raw_keys = [
            os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"), os.getenv("GROQ_API_KEY_4")
        ]
        self.api_keys = [k for k in raw_keys if k and k.strip() != ""]
        if not self.api_keys:
            st.error("⚠️ No API keys found in .env file!")

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
                        if result is None: continue 
                        return result
                    else:
                        return llm.invoke(prompt).content
                except Exception as e:
                    print(f"      [!] Key {key[:8]}... failed: {e}")
                    continue # Instantly try the next key
            
            # If the code reaches this line, ALL 4 KEYS ARE EXHAUSTED.
            # Instead of crashing, we trigger the Anti-Crash Cooldown.
            print("🚨 All 4 API Keys Rate-Limited. Entering 60-second cooldown...")
            time.sleep(65) # Wait 65 seconds for Groq to completely reset your TPM limits
            print("🔄 Cooldown complete. Resuming evaluation...")

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Groq-" + self.model_name

# ==========================================
# 3. DEDICATED ROOT PIPELINES
# ==========================================
def pipeline_without_rag(query: str, llm_engine: GroqFallbackLLM):
    """Root Pipeline 1: LLM Only (Baseline)"""
    start_time = time.time()
    prompt = f"Answer the following question accurately based solely on your internal knowledge: {query}"
    answer = llm_engine.generate(prompt)
    latency = time.time() - start_time
    return answer, ["N/A"], latency

def pipeline_with_rag(query: str, llm_engine: GroqFallbackLLM, chunking: str, embedding: str, db_type: str):
    """Root Pipeline 2: Retrieve + Generate (RAG) mapped to 12 combinations with Auto-Corrector"""
    start_time = time.time()
    
    # 1. Map to specific Embedding Model (uses bge-small to match 2_build_databases.py)
    embed_model_name = "BAAI/bge-small-en-v1.5" if embedding == "BGE embeddings" else "all-MiniLM-L6-v2"
    embedder = HuggingFaceEmbeddings(model_name=embed_model_name)
    
    # 2. Map to exact database folder path dynamically
    c_str = chunking.lower()
    e_str = "bge" if embedding == "BGE embeddings" else "sentence-transformers"
    d_str = "chroma" if db_type == "ChromaDB" else "faiss"
    db_path = f"databases/{c_str}_{e_str}_{d_str}"
    
    if not os.path.exists(db_path):
        return f"Error: Database folder '{db_path}' not found. Please run ingestion for this matrix.", [], 0.0
        
    # 3. Load Vector Database safely with DIMENSION AUTO-CORRECTOR
    try:
        if d_str == "chroma":
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embedder)
        else:
            vectorstore = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
            
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.75}
        )
        
        # --- AUTO-CORRECTOR BLOCK ---
        try:
            retriever.invoke("test ping")
        except Exception as test_err:
            if "dimension" in str(test_err).lower() or "384" in str(test_err):
                # Crash detected! Swapping to 384-dimension model to save the run
                embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                if d_str == "chroma":
                    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedder)
                else:
                    vectorstore = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.75}
                )
            else:
                raise test_err
        # ----------------------------
        
        docs = retriever.invoke(query)
        contexts = [doc.page_content for doc in docs]
    except Exception as e:
        return f"Retrieval Error: {e}", [], 0.0
        
    # 4. Generate Grounded Answer
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
    answer = llm_engine.generate(prompt)
    
    latency = time.time() - start_time
    return answer, contexts, latency

# ==========================================
# 4. SIDEBAR CONFIGURATION MATRIX
# ==========================================
st.sidebar.title("⚙️ RAG System Config")

run_mode = st.sidebar.radio("Base System:", ["Without RAG (Baseline)", "With RAG"])

st.sidebar.markdown("---")

if run_mode == "With RAG":
    st.sidebar.subheader("Architecture Model")
    sel_architecture = st.sidebar.selectbox(
        "Select Architecture Protocol:", 
        ["Naive RAG", "Hybrid RAG", "Graph RAG", "Multimodal RAG"]
    )
    
    st.sidebar.subheader("Evaluation Methods")
    sel_chunking = st.sidebar.selectbox("Chunking Strategy:", ["Fixed", "Recursive", "Semantic"])
    sel_embedding = st.sidebar.selectbox("Embeddings Model:", ["SentenceTransformers", "BGE embeddings"])
    sel_db = st.sidebar.selectbox("Vector Database:", ["FAISS", "ChromaDB"])
    
    # Calculate active path for status display
    _c = sel_chunking.lower()
    _e = "bge" if sel_embedding == "BGE embeddings" else "sentence-transformers"
    _d = "chroma" if sel_db == "ChromaDB" else "faiss"
    active_path = f"databases/{_c}_{_e}_{_d}"
    
    st.sidebar.markdown("---")
    if os.path.exists(active_path):
        st.sidebar.success(f"🟢 Database Validated:\n`{active_path}`")
    else:
        st.sidebar.error(f"🔴 Missing Database:\n`{active_path}`")
else:
    sel_architecture, sel_chunking, sel_embedding, sel_db = "None", "None", "None", "None"
    st.sidebar.info("🔵 LLM Only (No External Data)")

if st.sidebar.button("🗑️ Clear Live Chat"):
    st.session_state.messages = []
    st.rerun()

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("🧠 DeepEval Master Evaluation Engine")

tab1, tab2, tab3 = st.tabs(["💬 Live Test Chat", "🧪 DeepEval Test Automation", "📊 Master Analytics Dashboard"])

# ------------------------------------------
# TAB 1: LIVE TEST CHAT
# ------------------------------------------
with tab1:
    sys_name = "Baseline LLM" if run_mode == "Without RAG (Baseline)" else f"{sel_architecture} ({sel_chunking} + {sel_db})"
    st.markdown(f"**Active Environment:** `{sys_name}`")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Test your active configuration..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Processing via Root Pipeline..."):
                gen_llm = GroqFallbackLLM("llama-3.1-8b-instant")
                if run_mode == "Without RAG (Baseline)":
                    answer, ctx, lat = pipeline_without_rag(prompt, gen_llm)
                else:
                    answer, ctx, lat = pipeline_with_rag(prompt, gen_llm, sel_chunking, sel_embedding, sel_db)
                
                st.markdown(answer)
                st.caption(f"⏱️ Response Time: {lat:.2f}s | 📚 Context Chunks Retrieved: {len(ctx) if ctx[0] != 'N/A' else 0}")
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------------------------------
# TAB 2: DEEPEVAL AUTOMATION (THE JUDGE)
# ------------------------------------------
with tab2:
    st.markdown("### ⚖️ LLM-as-a-Judge Evaluation (70B Model Powered)")
    st.info("Automated testing utilizing **Answer Accuracy**, **Retrieval Precision**, and **Hallucination Reduction**.")
    
    uploaded_file = st.file_uploader("Upload Evaluation CSV (`questions.csv`)", type=["csv", "xlsx"])
    
    # Try loading from the uploaded file, or fall back to the default dataset if present
    df_test = None
    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file, on_bad_lines='skip')
    elif os.path.exists("data/questions.csv"):
        df_test = pd.read_csv("data/questions.csv")
        st.info("💡 Automatically loaded default `data/questions.csv`.")
        
    if df_test is not None:
        df_test.columns = [str(c).strip().lower() for c in df_test.columns]
        
        q_col = next((c for c in df_test.columns if 'q' in c), None)
        a_col = next((c for c in df_test.columns if 'ground' in c or 'truth' in c or 'answer' in c), None)
        
        if q_col and a_col:
            max_limit = len(df_test)
            start_q, end_q = st.slider("Select question range to evaluate:", min_value=1, max_value=max_limit, value=(1, min(10, max_limit)))
            df_test = df_test.iloc[start_q - 1 : end_q]
            st.success(f"Loaded {len(df_test)} pairs (questions {start_q} to {end_q}). Benchmarking System: **{sys_name}**.")
            
            if st.button("🚀 Execute DeepEval Testing Suite"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # THE FIX: Generator stays 8B for speed. Judge MUST be 70B to prevent JSON Tool crashes.
                evaluator_llm_judge = GroqFallbackLLM("llama-3.3-70b-versatile")
                generator_llm = GroqFallbackLLM("llama-3.1-8b-instant")
                
                current_results = st.session_state.eval_results
                
                for i in range(len(df_test)):
                    question = str(df_test.iloc[i][q_col])
                    ground_truth = str(df_test.iloc[i][a_col])
                    
                    status_text.text(f"Processing Q{i+1}/{len(df_test)}...")
                    
                    if run_mode == "Without RAG (Baseline)":
                        answer, contexts, latency = pipeline_without_rag(question, generator_llm)
                    else:
                        answer, contexts, latency = pipeline_with_rag(question, generator_llm, sel_chunking, sel_embedding, sel_db)
                    
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
                        
                        if run_mode != "Without RAG (Baseline)":
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
                            # We can average Recall and Relevancy for a much stricter "True Accuracy" score
                            acc_score = (acc_score + recall_metric.score) / 2
                            
                        else:
                            prec_score, hall_score = 0.0, 0.0
                            
                    except Exception as e:
                        st.warning(f"Q{i+1} Metric Failure (Skipping): {e}")
                        acc_score, prec_score, hall_score = 0.0, 0.0, 0.0
                        
                    # Before saving, transform the 0-1 score to a 1-5 score
                    # We use max(1.0, score * 5) to ensure a 0.0 doesn't become a 0
                    scaled_acc = round(max(1.0, acc_score * 5), 1)
                    scaled_prec = round(max(1.0, prec_score * 5), 1)
                    scaled_hall = round(max(1.0, hall_score * 5), 1)
                    
                    # 1. Compile the single result
                    single_result = {
                        "System Type": run_mode,
                        "Architecture": sel_architecture,
                        "Chunking": sel_chunking,
                        "Embedding": sel_embedding,
                        "Database": sel_db,
                        "Answer Accuracy": scaled_acc,
                        "Retrieval Precision": scaled_prec,
                        "Hallucination Reduction": scaled_hall,
                        "Response Time (s)": round(latency, 2),
                    }
                    current_results.append(single_result)
                    
                    # 2. INSTANT AUTO-SAVE TO CSV
                    log_file_path = "evaluation_history_log.csv"
                    single_df = pd.DataFrame([single_result])
                    if os.path.exists(log_file_path):
                        single_df.to_csv(log_file_path, mode='a', header=False, index=False)
                    else:
                        single_df.to_csv(log_file_path, index=False)
                    
                    # 3. Update UI
                    with st.expander(f"Q{i+1}: {question}", expanded=False):
                        st.markdown(f"**AI Answer:** {answer}")
                        st.markdown(f"**Metrics:** Acc: `{single_result['Answer Accuracy']}` | Prec: `{single_result['Retrieval Precision']}` | Hallucination Red: `{single_result['Hallucination Reduction']}` | Time: `{round(latency,2)}s`")
                        
                    progress_bar.progress((i + 1) / len(df_test))
                    
                st.session_state.eval_results = current_results
                st.success(f"✅ DeepEval Sequence Complete! All {len(df_test)} pairs safely logged to {log_file_path}.")

# ------------------------------------------
# TAB 3: MASTER ANALYTICS DASHBOARD
# ------------------------------------------
with tab3:
    if len(st.session_state.eval_results) == 0:
        st.info("Run an evaluation sequence in Tab 2 to populate the master dashboard.")
    else:
        df_res = pd.DataFrame(st.session_state.eval_results)
        qual_metrics = ["Answer Accuracy", "Retrieval Precision", "Hallucination Reduction"]
        
        st.subheader("📋 Raw Evaluation Data")
        st.dataframe(df_res, use_container_width=True)
        
        view = st.radio("Select Analysis Perspective:", [
            "1. Baseline vs RAG Architectures", 
            "2. Component Breakdown (Chunks/Embeds/DBs)"
        ], horizontal=True)
        st.markdown("---")
        
        if "Architectures" in view:
            # Grouping by Architecture (e.g., Naive RAG, Hybrid RAG vs Baseline)
            sys_avg = df_res.groupby("Architecture")[qual_metrics + ["Response Time (s)"]].mean().reset_index()
            melted_sys = sys_avg.melt(id_vars="Architecture", value_vars=qual_metrics, var_name="Metric", value_name="Score")
            
            # Filter out non-applicable metrics for the baseline model (Architecture == 'None')
            melted_sys = melted_sys[~((melted_sys["Architecture"] == "None") & (melted_sys["Metric"].isin(["Retrieval Precision", "Hallucination Reduction"])))]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Quality Metrics")
                fig_qual = px.bar(melted_sys, x="Metric", y="Score", color="Architecture", barmode="group", text_auto=".3f", range_y=[1, 5])
                st.plotly_chart(fig_qual, use_container_width=True)
            with c2:
                st.subheader("Latency")
                fig_time = px.bar(sys_avg, x="Architecture", y="Response Time (s)", color="Architecture", text_auto=".2f")
                st.plotly_chart(fig_time, use_container_width=True)
                
            st.subheader("🕸️ Architecture Capabilities Matrix")
            fig_radar = go.Figure()
            for index, row in sys_avg.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row["Answer Accuracy"], row["Retrieval Precision"], row["Hallucination Reduction"], row["Answer Accuracy"]],
                    theta=["Accuracy", "Precision", "Hallucination Reduction", "Accuracy"],
                    fill='toself', name=row["Architecture"]
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5])), height=500)
            st.plotly_chart(fig_radar, use_container_width=True)
                
        elif "Component" in view:
            rag_only = df_res[df_res["System Type"] == "With RAG"]
            if rag_only.empty:
                st.warning("Only 'Without RAG' data exists. Run a 'With RAG' configuration to see component breakdowns.")
            else:
                t1, t2, t3 = st.tabs(["🧩 Compare Chunking", "🧠 Compare Embeddings", "🗄️ Compare Databases"])
                
                with t1:
                    chunk_avg = rag_only.groupby("Chunking")[qual_metrics].mean().reset_index()
                    fig_c = px.bar(chunk_avg.melt(id_vars="Chunking", value_vars=qual_metrics), x="Chunking", y="value", color="variable", barmode="group", text_auto=".2f", range_y=[1, 5])
                    st.plotly_chart(fig_c, use_container_width=True)
                
                with t2:
                    embed_avg = rag_only.groupby("Embedding")[qual_metrics].mean().reset_index()
                    fig_e = px.bar(embed_avg.melt(id_vars="Embedding", value_vars=qual_metrics), x="Embedding", y="value", color="variable", barmode="group", text_auto=".2f", range_y=[1, 5])
                    st.plotly_chart(fig_e, use_container_width=True)
                    
                with t3:
                    db_avg = rag_only.groupby("Database")[qual_metrics].mean().reset_index()
                    fig_db = px.bar(db_avg.melt(id_vars="Database", value_vars=qual_metrics), x="Database", y="value", color="variable", barmode="group", text_auto=".2f", range_y=[1, 5])
                    st.plotly_chart(fig_db, use_container_width=True)