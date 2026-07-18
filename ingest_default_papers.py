import os
import urllib.request
from chains.rag_pipeline import RAGPipeline
from custom_logging.logger import app_logger
from config.settings import settings

def main():
    papers_dir = "data/papers"
    os.makedirs(papers_dir, exist_ok=True)
    
    # 1. Download Paper: Attention Is All You Need
    paper_url = "https://arxiv.org/pdf/1706.03762.pdf"
    paper_path = os.path.join(papers_dir, "attention_is_all_you_need.pdf")
    
    if not os.path.exists(paper_path):
        app_logger.info(f"Downloading paper from {paper_url}...")
        try:
            req = urllib.request.Request(paper_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(paper_path, 'wb') as out_file:
                out_file.write(response.read())
            app_logger.info("Download complete.")
        except Exception as e:
            app_logger.error(f"Failed to download {paper_url}: {e}")
            return
    else:
        app_logger.info(f"Paper {paper_path} already exists.")
        
    app_logger.info("Initializing RAGPipeline...")
    pipeline = RAGPipeline()
    
    app_logger.info("Ingesting paper into Weaviate vector database...")
    # Use huggingface as the default embedding model 
    result = pipeline.ingest_paper(
        file_path=paper_path,
        chunking_strategy="recursive",
        embedding_model_type=settings.get("embeddings", "default_model", default="huggingface"),
        namespace="default"
    )
    
    app_logger.info(f"Ingestion complete: {result}")

if __name__ == "__main__":
    main()
