import streamlit as st
import pandas as pd
import time
import os
import json
import random
from dotenv import load_dotenv

# --- AI & LangChain Imports ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

# Robust .env loading relative to script path
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)

st.set_page_config(page_title="Dataset Generator UI", layout="centered")

# ==========================================
# 1. THE 4-KEY API FALLBACK ENGINE
# ==========================================
class GroqFallbackGenerator:
    """Rotates 4 Groq API keys to bypass rate limits during massive generation runs."""
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        raw_keys = [
            os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"), os.getenv("GROQ_API_KEY_4")
        ]
        self.api_keys = [k for k in raw_keys if k and k.strip() != ""]
        if not self.api_keys:
            st.error("⚠️ No API keys found in .env file!")

    def invoke_json(self, prompt: str):
        for key in self.api_keys:
            try:
                llm = ChatGroq(api_key=key, model_name=self.model_name, temperature=0.2, max_retries=1)
                messages = [
                    SystemMessage(content="You must output ONLY raw, valid JSON. No markdown wrappers, no intro text."), 
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages)
                
                # Clean and parse JSON
                clean_text = response.content.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_text)
            except Exception as e:
                # Silently fail and rotate to the next key on 429/400 errors
                continue
                
        raise Exception("API Exhaustion: All keys rate-limited or JSON hallucinated.")

# ==========================================
# 2. DOCUMENT PROCESSING
# ==========================================
@st.cache_data
def load_and_chunk_pdfs():
    """Loads PDFs from /data and chunks them for question generation."""
    try:
        loader = PyPDFDirectoryLoader("data")
        documents = loader.load()
        if not documents:
            return None, "Error: No PDF files found in the 'data/' folder."
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        
        # Scrub invisible/corrupted characters that might break the JSON
        clean_chunks = []
        for chunk in chunks:
            clean_text = str(chunk.page_content).replace('\x00', '').strip()
            if len(clean_text) > 50: # Only keep substantial chunks
                chunk.page_content = clean_text
                clean_chunks.append(chunk)
                
        return clean_chunks, f"Successfully loaded and cleaned {len(clean_chunks)} chunks."
    except Exception as e:
        return None, f"Loader Error: {e}"

# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================
st.title("🛠️ Evaluation Dataset Generator")
st.markdown("Automatically generate highly precise Question/Ground Truth pairs from your `/data` PDFs. Optimized for DeepEval's strict accuracy metrics.")

# UI Controls
num_questions = st.slider("Target Number of Questions:", min_value=10, max_value=250, value=50, step=10)
st.info(f"Targeting **{num_questions}** total questions. API keys will rotate automatically to prevent rate limits.")

if st.button("🚀 Generate Golden Dataset"):
    chunks, status_msg = load_and_chunk_pdfs()
    
    if not chunks:
        st.error(status_msg)
    else:
        st.success(status_msg)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Shuffle chunks to ensure diverse questions across all documents
        random.seed(42)
        random.shuffle(chunks)
        
        generator = GroqFallbackGenerator("llama-3.1-8b-instant")
        dataset = []
        consecutive_failures = 0
        
        # Iterate through chunks until we hit the target number of questions
        for idx, chunk in enumerate(chunks):
            if len(dataset) >= num_questions:
                break
                
            status_text.text(f"Extracting Q&A Pair {len(dataset)+1}/{num_questions}...")
            
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
                
                # Validate JSON structure before saving
                if "question" in qa_pair and "ground_truth" in qa_pair:
                    dataset.append({
                        "Question": qa_pair["question"],
                        "Ground Truth": qa_pair["ground_truth"]
                    })
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                # Update UI
                progress_bar.progress(len(dataset) / num_questions)
                time.sleep(1) # Cooldown
                
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    st.error("API Exhaustion or severe JSON formatting issues. Stopping generation to save progress.")
                    break
                time.sleep(2) # Extended cooldown on error
                
        # Save directly to CSV for the Master App to read
        if dataset:
            df = pd.DataFrame(dataset)
            os.makedirs("data", exist_ok=True)
            csv_path = "data/questions.csv"
            df.to_csv(csv_path, index=False)
            
            st.success(f"✅ Master Dataset Completed! {len(dataset)} precise pairs saved to `{csv_path}`.")
            st.dataframe(df.head(), use_container_width=True)
            st.info("You can now open `3_evaluation_dashboard.py` and upload this CSV to run your evaluations!")
        else:
            st.error("Failed to generate any valid questions. Check API keys and limits.")