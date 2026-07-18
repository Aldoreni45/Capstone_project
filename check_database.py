"""
Quick diagnostic script to check if papers are indexed in Weaviate
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from vectordb.weaviate_client import WeaviateVectorClient
from config.settings import settings

def check_database():
    print("=" * 60)
    print("RAG Database Diagnostic Tool")
    print("=" * 60)
    
    try:
        # Connect to Weaviate
        print("\n🔌 Connecting to Weaviate...")
        client = WeaviateVectorClient('bge')
        
        # Check indexed papers
        print("📊 Checking indexed papers...")
        papers = client.get_ingested_papers()
        
        print(f"\n✅ Found {len(papers)} indexed papers:")
        
        if papers:
            for i, paper in enumerate(papers[:10], 1):
                print(f"  {i}. {paper.get('title', 'Unknown')} - {paper.get('source', 'Unknown')}")
            if len(papers) > 10:
                print(f"  ... and {len(papers) - 10} more papers")
        else:
            print("  ⚠️  No papers indexed! You need to upload PDFs first.")
            print("\n💡 To fix this:")
            print("  1. Open the Streamlit app")
            print("  2. Go to 'Document Ingest & Corpus' tab")
            print("  3. Upload a PDF research paper")
            print("  4. Click 'Index PDF to Vector Store'")
        
        client.close()
        
        print("\n" + "=" * 60)
        return len(papers) > 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("💡 Check your Weaviate credentials in .env file")
        return False

if __name__ == "__main__":
    has_papers = check_database()
    sys.exit(0 if has_papers else 1)
