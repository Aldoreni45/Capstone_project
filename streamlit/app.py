import os
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Adjust path and import settings
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from chains.rag_pipeline import RAGPipeline
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticTextChunker
from embeddings import get_embeddings, EmbeddingBenchmarker
from evaluation.evaluator import UnifiedRAGEvaluator
from custom_logging.logger import app_logger
from pydantic_models.responses import StructuredAnswer

# Page Configuration
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #00FFA3 0%, #00B8FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        color: #00FFA3;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = UnifiedRAGEvaluator()
if "uploaded_papers" not in st.session_state:
    try:
        from vectordb.weaviate_client import WeaviateVectorClient
        emb_type = settings.get("embeddings", "default_model", default="huggingface")
        vector_client = WeaviateVectorClient(emb_type)
        st.session_state.uploaded_papers = vector_client.get_ingested_papers()
        vector_client.close()
    except Exception as e:
        app_logger.warning(f"Could not load pre-ingested papers from Weaviate: {e}")
        st.session_state.uploaded_papers = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{int(datetime.now().timestamp())}"
if "last_results" not in st.session_state:
    st.session_state.last_results = None  # Stores (answer, top_chunks, metrics, evaluation_output)
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

pipeline = st.session_state.pipeline
evaluator = st.session_state.evaluator

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.markdown("<h2 style='color:#00FFA3; margin-top:0;'>Settings Engine</h2>", unsafe_allow_html=True)
    
    st.divider()

    # 2. Pipeline settings
    st.markdown("### 🎛️ Pipeline Architectures")
    
    emb_selection = st.selectbox(
        "Embedding Model",
        options=["huggingface", "bge"],
        index=0 if settings.get("embeddings", "default_model") == "huggingface" else 1,
        help="HuggingFace (bge-large-en-v1.5) or BGE small — both run locally, no API key required."
    )
    
    retriever_selection = st.selectbox(
        "Retrieval Strategy",
        options=["cosine", "mmr", "hybrid", "multiquery", "parent", "compression", "ensemble", "selfquery", "threshold"],
        index=2,  # Defaults to Hybrid Search
        help="Select the retrieval engine. Hybrid fuses dense and sparse BM25 results."
    )
    
    llm_selection = st.selectbox(
        "Groq LLM Model",
        options=[
            settings.get("llm", "groq", "supported_models", "llama_3_3_70b"),
            settings.get("llm", "groq", "supported_models", "deepseek_r1_70b"),
            settings.get("llm", "groq", "supported_models", "gemma_2_9b"),
            settings.get("llm", "groq", "supported_models", "qwen_2_5_coder"),
            settings.get("llm", "groq", "supported_models", "mixtral_8x7b")
        ],
        index=0,
        help="Select the inference model from Groq Cloud."
    )
    
    memory_selection = st.selectbox(
        "Conversational Memory",
        options=["buffer", "token", "summary"],
        index=0,
        help="Prunes or summarizes history contexts automatically."
    )

    st.divider()

    # 3. Chunking Settings
    st.markdown("### ✂️ Chunking Strategies")
    chunk_strategy = st.radio("Active Chunker", options=["recursive", "semantic"], index=0)
    
    # 4. Diagnostics & Controls
    st.markdown("### 🛠️ Diagnostics")
    if st.button("🔄 Reset Chat & Memory", type="primary", width='stretch'):
        st.session_state.chat_messages = []
        st.session_state.last_results = None
        pipeline.memory_manager.clear_session(st.session_state.session_id)
        st.success("Session memory wiped.")
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN APP INTERFACE & TABS
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>Research Paper Answer Bot</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Production-Grade Retrieval-Augmented Generation (RAG) Architecture</div>", unsafe_allow_html=True)

tab_chat, tab_docs, tab_metrics, tab_eval = st.tabs([
    "💬 Chat Assistant",
    "📚 Document Ingest & Corpus",
    "📊 Performance Metrics",
    "🎯 RAG Evaluations"
])

# -----------------------------------------------------------------------------
# TAB 1: CHAT ASSISTANT
# -----------------------------------------------------------------------------
with tab_chat:
    if not st.session_state.uploaded_papers:
        st.info("💡 Note: No new papers uploaded in this session. Querying pre-ingested papers in the vector database.")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### Chat History")
        
        # Display chat bubbles
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Query input
        user_query = st.chat_input("Ask a question about the papers...")
        
        if user_query:
            # Check requirements
            if not settings.groq_api_key:
                st.error("Please enter a Groq API Key in the sidebar expander first.")
            else:
                # Add to UI
                st.session_state.chat_messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)
                    
                # Execute pipeline query
                with st.chat_message("assistant"):
                    with st.spinner("Processing your query..."):
                        try:
                            result = pipeline.answer_query(
                                query=user_query,
                                session_id=st.session_state.session_id,
                                retriever_type=retriever_selection,
                                embedding_model_type=emb_selection,
                                llm_model=llm_selection,
                                memory_type=memory_selection,
                                namespace="default"
                            )
                            
                            # Handle different return types
                            # BAD retrieval returns dict, GOOD/PARTIAL returns tuple
                            if isinstance(result, dict):
                                # BAD retrieval - graceful failure message
                                ans_struct = StructuredAnswer(
                                    answer=result["answer"],
                                    citations=[],
                                    confidence_score=result["confidence"]
                                )
                                top_chunks = []
                                pipeline_metrics = {
                                    "retrieval_quality": result["retrieval_quality"],
                                    "retrieval_reason": result["retrieval_reason"],
                                    "query_rewritten": result["query_rewritten"],
                                    "web_search_used": result["web_search_used"]
                                }
                            else:
                                # GOOD/PARTIAL retrieval - normal tuple
                                ans_struct, top_chunks, pipeline_metrics = result
                            
                            # Display CRAG status indicators
                            crag_status_container = st.container()
                            with crag_status_container:
                                if not top_chunks:
                                    st.info("💬 **General Knowledge Response**")
                                else:
                                    # Check if web search was used (from metrics)
                                    web_used = pipeline_metrics.get("web_search_used", False)
                                    query_rewritten = pipeline_metrics.get("query_rewritten", False)
                                    retrieval_quality = pipeline_metrics.get("retrieval_quality", "GOOD")
                                    
                                    status_parts = []
                                    if query_rewritten:
                                        status_parts.append("🔄 Query Rewritten")
                                    if web_used:
                                        status_parts.append("🌐 Web Search Enhanced")
                                    
                                    if status_parts:
                                        st.info(f"📚 **Document-Based Response** ({' + '.join(status_parts)})")
                                    else:
                                        st.info("📚 **Document-Based Response**")
                                    
                                    # Show retrieval quality badge
                                    quality_colors = {
                                        "GOOD": "#00FFA3",
                                        "PARTIAL": "#FFA500", 
                                        "BAD": "#FF4B4B"
                                    }
                                    quality_color = quality_colors.get(retrieval_quality, "#00FFA3")
                                    st.markdown(
                                        f"<small style='color:{quality_color}; font-weight:bold;'>"
                                        f"Retrieval Quality: {retrieval_quality}"
                                        f"</small>",
                                        unsafe_allow_html=True
                                    )
                            
                            # Display answer
                            st.write(ans_struct.answer)
                            
                            # Display citations if available
                            if ans_struct.citations:
                                st.divider()
                                st.markdown("**📖 Sources:**")
                                for idx, citation in enumerate(ans_struct.citations, 1):
                                    with st.expander(f"Source {idx}: {citation.title}", expanded=False):
                                        st.markdown(f"**Page:** {citation.page}")
                                        st.markdown(f"**Chunk ID:** {citation.chunk_id}")
                                        st.markdown(f"**Passage:** {citation.passage}")
                            
                            # Display confidence level
                            st.caption(f"Confidence: {ans_struct.confidence_score:.1%}")
                            
                            # Log to chat messages state
                            st.session_state.chat_messages.append({"role": "assistant", "content": ans_struct.answer})
                            
                            # Execute dynamic evaluation using the evaluator
                            # Skip evaluation if retrieval quality is BAD
                            if pipeline_metrics.get("retrieval_quality") != "BAD":
                                eval_out = evaluator.evaluate_turn(
                                    query=user_query,
                                    retrieved_chunks=top_chunks,
                                    actual_answer=ans_struct.answer
                                )
                            else:
                                # Create empty evaluation for BAD retrievals
                                eval_out = {
                                    'faithfulness': 0.0,
                                    'context_precision': 0.0,
                                    'context_recall': 0.0,
                                    'hallucination_score': 1.0,
                                    'answer_relevancy': 0.0,
                                    'semantic_similarity': 0.0,
                                    'retrieval_accuracy': 0.0,
                                    'verdict': 'Evaluation skipped due to BAD retrieval quality'
                                }
                            
                            # Cache last result details
                            st.session_state.last_results = {
                                "answer": ans_struct,
                                "chunks": top_chunks,
                                "metrics": pipeline_metrics,
                                "eval": eval_out
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error executing RAG pipeline: {str(e)}")
                            app_logger.error(f"UI Error during RAG pipeline query: {str(e)}")

    with col_right:
        st.markdown("### Context & Citations")
        if st.session_state.last_results:
            results = st.session_state.last_results
            ans = results["answer"]
            chunks = results["chunks"]
            eval_data = results["eval"]
            
            # Handle both dict and object types for eval_data
            if isinstance(eval_data, dict):
                faithfulness = eval_data.get('faithfulness', 0.0)
                context_precision = eval_data.get('context_precision', 0.0)
            else:
                faithfulness = getattr(eval_data, 'faithfulness', 0.0)
                context_precision = getattr(eval_data, 'context_precision', 0.0)
            
            # Confidence score display
            st.markdown(f"#### Confidence Score")
            score_color = "#00FFA3" if ans.confidence_score >= 0.7 else "#FFA500" if ans.confidence_score >= 0.4 else "#FF4B4B"
            st.markdown(f"<div style='font-size:2.2rem; color:{score_color}; font-weight:800;'>{ans.confidence_score * 100:.1f}%</div>", unsafe_allow_html=True)
            st.progress(ans.confidence_score)
            
            st.divider()

            # Cite references
            st.markdown("#### Top Cited Sources")
            if ans.citations:
                for i, cite in enumerate(ans.citations[:3]):
                    with st.expander(f"Reference {i+1}: {cite.title} - Page {cite.page}", expanded=(i==0)):
                        st.markdown(f"**Chunk ID**: `{cite.chunk_id}`")
                        st.markdown(f"**Source File**: `{cite.source}`")
                        st.markdown(f"*Cited Passage*:\n> {cite.passage}")
            else:
                st.info("No citations available (retrieval quality was BAD)")

            st.divider()
            
            # Download actions
            st.markdown("#### Actions")
            md_output = f"""# Research Paper Bot Answer
**Query**: {user_query if 'user_query' in locals() else 'Last Query'}
**Confidence Score**: {ans.confidence_score * 100:.1f}%

## Answer
{ans.answer}

## Citations
"""
            for i, cite in enumerate(ans.citations):
                md_output += f"\n### Citation {i+1}\n* Paper: {cite.title}\n* Page: {cite.page}\n* Chunk ID: {cite.chunk_id}\n* Passage: {cite.passage}\n"
                
            st.download_button(
                label="📥 Download Answer Markdown",
                data=md_output,
                file_name="rag_answer.md",
                mime="text/markdown",
                width='stretch'
            )
        else:
            st.write("Submit a question to view citations, confidence ratings, and export options.")

# -----------------------------------------------------------------------------
# TAB 2: DOCUMENT INGEST & CORPUS
# -----------------------------------------------------------------------------
with tab_docs:
    st.markdown("### Document Upload & Persistent Ingestion")
    
    col_up_left, col_up_right = st.columns([1, 1])
    
    with col_up_left:
        uploaded_file = st.file_uploader("Choose a PDF research paper", type="pdf")
        
        if uploaded_file is not None:
            # Setup namespace
            ns_input = st.text_input("Ingest Namespace", value="default")
            
            if st.button("⚡ Index PDF to Vector Store", width='stretch'):
                # Save stream to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    with st.spinner("Processing PDF loaders, running chunker, generating embeddings, and upserting vectors..."):
                        stats = pipeline.ingest_paper(
                            file_path=tmp_path,
                            chunking_strategy=chunk_strategy,
                            embedding_model_type=emb_selection,
                            namespace=ns_input
                        )
                        
                        meta = stats["metadata"]
                        st.session_state.uploaded_papers.append({
                            "title": meta.get("paper_title", uploaded_file.name),
                            "author": meta.get("author", "Unknown"),
                            "source": uploaded_file.name,
                            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "chunk_count": stats["chunk_count"]
                        })
                        st.success(f"Successfully indexed '{uploaded_file.name}' into namespace '{ns_input}'! Generated {stats['chunk_count']} chunks.")
                except Exception as e:
                    st.error(f"Failed to ingest paper: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    with col_up_right:
        st.markdown("### Active Chunking Benchmarking Analysis")
        st.markdown("Evaluate splits between **Recursive Text Splitter** and **Semantic Similarity Splitter**.")
        
        if uploaded_file is not None and st.button("📊 Compare Splitters", width='stretch'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                # Load pages
                pages = pipeline.loader.load_paper(tmp_path)
                
                # Split using both
                rec_split = RecursiveChunker()
                chunks_rec = rec_split.split_documents(pages)
                
                emb_model = get_embeddings(emb_selection)
                sem_split = SemanticTextChunker(emb_model)
                chunks_sem = sem_split.split_documents(pages)
                
                # Get metrics
                rec_lengths = [len(c.page_content) for c in chunks_rec]
                sem_lengths = [len(c.page_content) for c in chunks_sem]
                
                df_compare = pd.DataFrame({
                    "Metric": ["Chunk Count", "Average Length (char)", "Max Length", "Min Length"],
                    "Recursive Chunker": [len(chunks_rec), int(pd.Series(rec_lengths).mean()), max(rec_lengths), min(rec_lengths)],
                    "Semantic Chunker": [len(chunks_sem), int(pd.Series(sem_lengths).mean()) if sem_lengths else 0, max(sem_lengths) if sem_lengths else 0, min(sem_lengths) if sem_lengths else 0]
                })
                
                st.table(df_compare)
                
                # Histograms
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=rec_lengths, name="Recursive Splits", marker_color="#00B8FF", opacity=0.75))
                fig.add_trace(go.Histogram(x=sem_lengths, name="Semantic Splits", marker_color="#00FFA3", opacity=0.75))
                fig.update_layout(
                    title_text="Chunk Length distribution comparison",
                    xaxis_title_text="Length (characters)",
                    yaxis_title_text="Count",
                    barmode="overlay",
                    paper_bgcolor="#1E293B",
                    plot_bgcolor="#1E293B",
                    font_color="#F8FAFC"
                )
                st.plotly_chart(fig, width='stretch')
                
            except Exception as e:
                st.error(f"Failed to benchmark: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    st.divider()
    
    st.markdown("### Indexed Corpus Directory")
    if st.session_state.uploaded_papers:
        df_corpus = pd.DataFrame(st.session_state.uploaded_papers)
        st.dataframe(df_corpus, width='stretch')
    else:
        st.write("No papers indexed yet.")

# -----------------------------------------------------------------------------
# TAB 3: PERFORMANCE METRICS
# -----------------------------------------------------------------------------
with tab_metrics:
    st.markdown("### System Latency & Token Consumption Metrics")
    
    if st.session_state.last_results:
        metrics = st.session_state.last_results.get("metrics", {})
        
        # Handle case where metrics might be incomplete (BAD retrieval)
        latencies = metrics.get("avg_latency_ms", {})
        tokens = metrics.get("tokens", {})
        
        # 1. Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{latencies.get('total_pipeline', 0.0):.1f} ms</div>
                    <div class='metric-label'>End-to-End Latency</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{latencies.get('retrieval', 0.0):.1f} ms</div>
                    <div class='metric-label'>Vector Retrieval Latency</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{latencies.get('llm_generation', 0.0):.1f} ms</div>
                    <div class='metric-label'>LLM Structured Output Latency</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{tokens.get('total_tokens', 0)}</div>
                    <div class='metric-label'>Total Session Tokens</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # 2. Charts
        col_chart_left, col_chart_right = st.columns(2)
        
        with col_chart_left:
            # Latency break downs
            latency_labels = ["Vector Retrieval", "Reranker Model", "LLM Inference", "Other Steps"]
            ret_val = latencies.get("retrieval", 0.0)
            rer_val = latencies.get("reranking", 0.0)
            llm_val = latencies.get("llm_generation", 0.0)
            total = latencies.get("total_pipeline", 0.0)
            other = max(total - (ret_val + rer_val + llm_val), 0.0)
            
            fig_lat = px.pie(
                values=[ret_val, rer_val, llm_val, other],
                names=latency_labels,
                color_discrete_sequence=["#00B8FF", "#8A5CF5", "#00FFA3", "#475569"],
                title="Latency Breakdown (ms)"
            )
            fig_lat.update_layout(
                paper_bgcolor="#1E293B",
                plot_bgcolor="#1E293B",
                font_color="#F8FAFC"
            )
            st.plotly_chart(fig_lat, width='stretch')
            
        with col_chart_right:
            # Tokens break downs
            fig_tok = go.Figure(data=[
                go.Bar(
                    name="Prompt Tokens",
                    x=["Session Token Usage"],
                    y=[tokens.get("prompt_tokens", 0)],
                    marker_color="#00B8FF"
                ),
                go.Bar(
                    name="Completion Tokens",
                    x=["Session Token Usage"],
                    y=[tokens.get("completion_tokens", 0)],
                    marker_color="#00FFA3"
                )
            ])
            fig_tok.update_layout(
                barmode="stack",
                title="Token Allocation",
                paper_bgcolor="#1E293B",
                plot_bgcolor="#1E293B",
                font_color="#F8FAFC"
            )
            st.plotly_chart(fig_tok, width='stretch')

    else:
        st.info("Submit a question to see token metrics and pipeline execution charts.")

# -----------------------------------------------------------------------------
# TAB 4: RAG EVALUATIONS
# -----------------------------------------------------------------------------
with tab_eval:
    st.markdown("### Unified RAGAS & DeepEval Quality Metrics Dashboard")
    
    if st.session_state.last_results:
        eval_data = st.session_state.last_results["eval"]
        
        # Handle both dict and object types for eval_data
        if isinstance(eval_data, dict):
            faithfulness = eval_data.get('faithfulness', 0.0)
            context_precision = eval_data.get('context_precision', 0.0)
            context_recall = eval_data.get('context_recall', 0.0)
            hallucination_score = eval_data.get('hallucination_score', 1.0)
            answer_relevancy = eval_data.get('answer_relevancy', 0.0)
            semantic_similarity = eval_data.get('semantic_similarity', 0.0)
            retrieval_accuracy = eval_data.get('retrieval_accuracy', 0.0)
            verdict = eval_data.get('verdict', 'No evaluation data')
        else:
            faithfulness = getattr(eval_data, 'faithfulness', 0.0)
            context_precision = getattr(eval_data, 'context_precision', 0.0)
            context_recall = getattr(eval_data, 'context_recall', 0.0)
            hallucination_score = getattr(eval_data, 'hallucination_score', 1.0)
            answer_relevancy = getattr(eval_data, 'answer_relevancy', 0.0)
            semantic_similarity = getattr(eval_data, 'semantic_similarity', 0.0)
            retrieval_accuracy = getattr(eval_data, 'retrieval_accuracy', 0.0)
            verdict = getattr(eval_data, 'verdict', 'No evaluation data')
        
        col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
        with col_ev1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{faithfulness * 100:.1f}%</div>
                    <div class='metric-label'>Faithfulness</div>
                </div>
            """, unsafe_allow_html=True)
        with col_ev2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{context_precision * 100:.1f}%</div>
                    <div class='metric-label'>Context Precision</div>
                </div>
            """, unsafe_allow_html=True)
        with col_ev3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{context_recall * 100:.1f}%</div>
                    <div class='metric-label'>Context Recall</div>
                </div>
            """, unsafe_allow_html=True)
        with col_ev4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{semantic_similarity * 100:.1f}%</div>
                    <div class='metric-label'>Semantic Similarity</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.divider()

        # Gauge Charts
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hallucination_score * 100,
                title={"text": "Hallucination Risk (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "#FF4B4B"},
                    "steps": [
                        {"range": [0, 30], "color": "#1E293B"},
                        {"range": [30, 70], "color": "#334155"},
                        {"range": [70, 100], "color": "#475569"}
                    ]
                }
            ))
            fig_g1.update_layout(paper_bgcolor="#1E293B", font_color="#F8FAFC")
            st.plotly_chart(fig_g1, width='stretch')
            
        with col_g2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=retrieval_accuracy * 100,
                title={"text": "Retrieval Target Accuracy (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "#00FFA3"},
                    "steps": [
                        {"range": [0, 50], "color": "#1E293B"},
                        {"range": [50, 100], "color": "#334155"}
                    ]
                }
            ))
            fig_g2.update_layout(paper_bgcolor="#1E293B", font_color="#F8FAFC")
            st.plotly_chart(fig_g2, width='stretch')

        st.info(f"📋 **Evaluator Verdict**: {verdict}")
        
        st.divider()
        
        # Interactive ground truth override
        st.markdown("#### Test Suite Benchmark Engine")
        custom_gt = st.text_area("Provide the 'Ground Truth' answer to calculate exact benchmark precision:", value="")
        if st.button("🔥 Re-run Evaluation with Ground Truth", width='stretch'):
            with st.spinner("Re-measuring faithfulness, precision, recall, and semantic similarity..."):
                last_res = st.session_state.last_results
                updated_eval = evaluator.evaluate_turn(
                    query=st.session_state.chat_messages[-2]["content"],
                    retrieved_chunks=last_res["chunks"],
                    actual_answer=last_res["answer"].answer,
                    ground_truth=custom_gt
                )
                st.session_state.last_results["eval"] = updated_eval
                st.success("Evaluations recalculated!")
                st.rerun()
    else:
        st.info("Submit a question to see real-time quality verification checks.")
