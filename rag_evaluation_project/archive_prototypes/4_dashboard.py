import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Ultimate RAG Evaluation Dashboard", layout="wide")

# ==========================================
# 1. DATA PARSER & LOADER
# ==========================================
@st.cache_data
def load_and_parse_data():
    file_path = "results/benchmark_scores.json"
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    with open(file_path, "r") as f:
        raw_data = json.load(f)
        
    parsed_data = []
    for config, scores in raw_data.items():
        if "Without_RAG" in config:
            parsed_data.append({
                "Configuration Name": "Baseline LLM",
                "System Type": "Without RAG",
                "Architecture": "Without RAG",
                "Chunking": "None",
                "Embedding": "None",
                "Database": "None",
                **scores
            })
        else:
            # Smart architecture detector based on folder name
            config_lower = config.lower()
            if "hybrid" in config_lower: arch = "Hybrid RAG"
            elif "graph" in config_lower: arch = "Graph RAG"
            elif "multimodal" in config_lower: arch = "Multimodal RAG"
            elif "naive" in config_lower: arch = "Naive RAG"
            else: arch = "Standard RAG" # Fallback if specific name wasn't used
            
            # Extract Components (assuming format like semantic_bge_chroma)
            parts = config.split("_")
            chunking = parts[0].capitalize() if len(parts) > 0 else "Unknown"
            embedding = parts[1].upper() if len(parts) > 1 else "Unknown"
            database = parts[2].capitalize() if len(parts) > 2 else "Unknown"

            parsed_data.append({
                "Configuration Name": config,
                "System Type": "With RAG",
                "Architecture": arch,
                "Chunking": chunking,
                "Embedding": embedding,
                "Database": database,
                **scores
            })
    return pd.DataFrame(parsed_data)

df = load_and_parse_data()

if df.empty:
    st.error("⚠️ Data not found. Please ensure 'results/benchmark_scores.json' is generated.")
    st.stop()

# Define the Professor's Rubric Metrics
quality_metrics = ["answer_accuracy", "retrieval_precision", "hallucination_reduction"]
all_metrics = quality_metrics + ["response_time"]

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📊 Evaluation Dashboard")
view_mode = st.sidebar.radio(
    "Select Analysis Module:",
    [
        "1. Overall: RAG vs Without RAG", 
        "2. Architecture Models vs Baseline", 
        "3. Internal Component Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Evaluation Rubric:**\n"
    "- **Answer Accuracy:** Match to Ground Truth\n"
    "- **Retrieval Precision:** Context Relevance\n"
    "- **Hallucination Reduction:** Factual Grounding\n"
    "- **Response Time:** System Latency (Seconds)"
)

# =========================================================
# VIEW 1: OVERALL RAG VS WITHOUT RAG
# =========================================================
if view_mode == "1. Overall: RAG vs Without RAG":
    st.title("⚖️ Overall Impact: RAG vs. Baseline LLM")
    st.markdown("Comparing the raw foundational LLM against the average performance of all RAG systems combined.")
    
    sys_avg = df.groupby("System Type")[all_metrics].mean().reset_index()
    melted_sys = sys_avg.melt(id_vars="System Type", value_vars=quality_metrics, var_name="Metric", value_name="Score")
    melted_sys["Metric"] = melted_sys["Metric"].str.replace("_", " ").str.title()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_bar = px.bar(
            melted_sys, x="Metric", y="Score", color="System Type", barmode="group",
            text_auto=".3f", color_discrete_sequence=["#EF553B", "#00CC96"]
        )
        fig_bar.update_layout(yaxis_title="Score (0.0 - 1.0)", xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.subheader("System Latency")
        fig_time = px.bar(
            sys_avg, x="System Type", y="response_time", color="System Type", 
            text_auto=".2f", color_discrete_sequence=["#EF553B", "#00CC96"]
        )
        fig_time.update_layout(yaxis_title="Seconds (Lower is Better)", showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)

# =========================================================
# VIEW 2: SPECIFIC ARCHITECTURES VS WITHOUT RAG
# =========================================================
elif view_mode == "2. Architecture Models vs Baseline":
    st.title("🏗️ Architecture Comparison")
    st.markdown("Evaluating Naive, Hybrid, Graph, and Multimodal RAG specifically against the Without RAG baseline.")
    
    # Get averages per architecture, including "Without RAG"
    arch_avg = df.groupby("Architecture")[all_metrics].mean().reset_index()
    
    # Spider Chart for Quality
    st.subheader("🕸️ Multidimensional Architecture Footprint")
    fig_radar = go.Figure()
    
    for index, row in arch_avg.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["answer_accuracy"], row["retrieval_precision"], row["hallucination_reduction"], row["answer_accuracy"]],
            theta=["Accuracy", "Precision", "Hallucination Reduction", "Accuracy"],
            fill='toself',
            name=row["Architecture"]
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Bar Chart for Direct Comparison
    st.subheader("📊 Direct Metric Comparison")
    melted_arch = arch_avg.melt(id_vars="Architecture", value_vars=quality_metrics, var_name="Metric", value_name="Score")
    melted_arch["Metric"] = melted_arch["Metric"].str.replace("_", " ").str.title()
    
    fig_arch_bar = px.bar(
        melted_arch, x="Architecture", y="Score", color="Metric", barmode="group", text_auto=".2f"
    )
    st.plotly_chart(fig_arch_bar, use_container_width=True)

# =========================================================
# VIEW 3: COMPONENT ANALYSIS
# =========================================================
elif view_mode == "3. Internal Component Analysis":
    st.title("⚙️ RAG Component Evaluation")
    st.markdown("Isolating how Chunking Strategies, Embedding Models, and Vector Databases impact the 4 primary metrics.")
    
    # Filter out the baseline, only analyze RAG components
    rag_only = df[df["System Type"] == "With RAG"]
    
    tab1, tab2, tab3 = st.tabs(["🧩 Chunking Optimization", "🧠 Embedding Models", "🗄️ Vector Databases"])
    
    with tab1:
        st.subheader("Impact of Chunking Strategy")
        chunk_avg = rag_only.groupby("Chunking")[all_metrics].mean().reset_index()
        melted_chunk = chunk_avg.melt(id_vars="Chunking", value_vars=quality_metrics, var_name="Metric", value_name="Score")
        fig_c = px.bar(melted_chunk, x="Chunking", y="Score", color="Metric", barmode="group", text_auto=".2f")
        st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(chunk_avg, use_container_width=True)
        
    with tab2:
        st.subheader("Impact of Embedding Models")
        embed_avg = rag_only.groupby("Embedding")[all_metrics].mean().reset_index()
        melted_embed = embed_avg.melt(id_vars="Embedding", value_vars=quality_metrics, var_name="Metric", value_name="Score")
        fig_e = px.bar(melted_embed, x="Embedding", y="Score", color="Metric", barmode="group", text_auto=".2f")
        st.plotly_chart(fig_e, use_container_width=True)
        st.dataframe(embed_avg, use_container_width=True)
        
    with tab3:
        st.subheader("Impact of Vector Databases")
        db_avg = rag_only.groupby("Database")[all_metrics].mean().reset_index()
        melted_db = db_avg.melt(id_vars="Database", value_vars=quality_metrics, var_name="Metric", value_name="Score")
        fig_db = px.bar(melted_db, x="Database", y="Score", color="Metric", barmode="group", text_auto=".2f")
        st.plotly_chart(fig_db, use_container_width=True)
        st.dataframe(db_avg, use_container_width=True)