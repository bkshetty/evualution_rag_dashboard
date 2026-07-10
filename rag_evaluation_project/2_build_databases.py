import os
import json
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document # Imported to reconstruct clean chunks

load_dotenv()

def load_raw_notes():
    print("[+] Loading your 36+ college notes PDFs...")
    loader = PyPDFDirectoryLoader("data")
    return loader.load()

def get_embedding_model(model_choice):
    """Dynamically loads local embedding models safely."""
    if model_choice == "sentence-transformers":
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif model_choice == "bge":
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    else:
        raise ValueError("Invalid embedding selection")

def chunk_documents(documents, method):
    """Executes the specific chunking strategy requested."""
    if method == "fixed":
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=" ")
        return splitter.split_documents(documents)
        
    elif method == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)
        
    elif method == "semantic":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_documents(documents)
    else:
        raise ValueError("Unknown chunking strategy")

def build_matrix():
    docs = load_raw_notes()
    if not docs:
        print("[-] Error: Ensure PDFs are located in the 'data/' directory.")
        return

    chunking_methods = ["fixed", "recursive", "semantic"]
    embedding_models = ["sentence-transformers", "bge"]
    vector_dbs = ["faiss", "chroma"]

    os.makedirs("databases", exist_ok=True)

    print(f"\n[+] Total configurations to build: {len(chunking_methods) * len(embedding_models) * len(vector_dbs)}")
    
    for chunk_style in chunking_methods:
        print(f"\n--- Splitting data using [{chunk_style.upper()}] method ---")
        raw_chunks = chunk_documents(docs, chunk_style)
        
        # ==========================================================
        # THE AGGRESSIVE SANITIZER
        # Destroys Null Bytes, cleans metadata, and rebuilds objects
        # ==========================================================
        chunks = []
        for chunk in raw_chunks:
            if not hasattr(chunk, 'page_content') or chunk.page_content is None:
                continue
                
            # 1. Force to string and strip
            clean_text = str(chunk.page_content).strip()
            
            # 2. Destroy hidden PDF Null Bytes and invisible characters
            clean_text = clean_text.replace('\x00', '')
            clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text) # Strip weird unicode
            
            # 3. Only keep chunks that actually have sentences (more than 10 characters)
            if len(clean_text) > 10:
                # 4. Physically reconstruct the Document object
                safe_metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                clean_doc = Document(page_content=clean_text, metadata=safe_metadata)
                chunks.append(clean_doc)
                
        print(f"Created {len(chunks)} pure chunks (Scrubbed {len(raw_chunks) - len(chunks)} corrupted blocks).")
        # ==========================================================

        for embed_style in embedding_models:
            print(f"    -> Loading [{embed_style}] embeddings into cache...")
            embeddings = get_embedding_model(embed_style)

            for db_style in vector_dbs:
                db_name = f"{chunk_style}_{embed_style}_{db_style}"
                db_path = os.path.join("databases", db_name)
                
                print(f"       [*] Compiling Vector Store: {db_name}")
                
                if db_style == "faiss":
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    vector_store.save_local(db_path)
                elif db_style == "chroma":
                    vector_store = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        persist_directory=db_path
                    )
                
                print(f"       [✓] Saved successfully to {db_path}")

    print("\n[✓][✓][✓] System Ingestion Complete! All matrix stores are baked and ready in /databases.")

if __name__ == "__main__":
    build_matrix()