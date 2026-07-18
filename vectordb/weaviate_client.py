import uuid
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery
from langchain_core.documents import Document
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import WeaviateAPIError, retry_on_exception

class WeaviateVectorClient:
    """Production-grade Weaviate v4 client managing schema creation and vector operations."""

    def __init__(self, embedding_model_type: str = "bge"):
        self.url = settings.weaviate_url
        self.api_key = settings.weaviate_api_key
        self.model_type = embedding_model_type.lower()
        
        self.collection_name = settings.get("weaviate", "collection_name", default="ResearchPaper")
        self._connect()
        self._ensure_collection_exists()

    def _connect(self):
        """Establishes connection to Weaviate cloud or local server."""
        try:
            if self.url and self.api_key:
                app_logger.info(f"Connecting to Weaviate Cloud at {self.url}...")
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=Auth.api_key(self.api_key),
                    skip_init_checks=True
                )
            else:
                host = settings.get("weaviate", "local_host", default="localhost")
                port = settings.get("weaviate", "local_port", default=8080)
                app_logger.info(f"Connecting to local Weaviate at {host}:{port}...")
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    skip_init_checks=True
                )
            
            if not self.client.is_ready():
                raise ConnectionError("Weaviate cluster is not ready.")
            app_logger.info("Successfully connected to Weaviate database.")
        except Exception as e:
            app_logger.error(f"Weaviate connection failed: {str(e)}")
            raise WeaviateAPIError(f"Weaviate connection failed: {str(e)}")

    def _ensure_collection_exists(self):
        """Creates the target collection schema if it does not already exist."""
        try:
            if not self.client.collections.exists(self.collection_name):
                app_logger.info(f"Creating Weaviate collection '{self.collection_name}' with custom properties...")
                self.client.collections.create(
                    name=self.collection_name,
                    # We supply our own embeddings, so we disable the default auto-vectorizer
                    vectorizer_config=None,
                    properties=[
                        Property(name="paper_title", data_type=DataType.TEXT),
                        Property(name="page_number", data_type=DataType.INT),
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="author", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="embedding_model", data_type=DataType.TEXT),
                        Property(name="namespace", data_type=DataType.TEXT),
                        Property(name="text", data_type=DataType.TEXT),
                    ]
                )
                app_logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                app_logger.debug(f"Collection '{self.collection_name}' exists.")
        except Exception as e:
            app_logger.error(f"Failed to create schema collection: {str(e)}")
            raise WeaviateAPIError(f"Weaviate schema creation failed: {str(e)}")

    def _generate_uuid(self, chunk_id: str) -> str:
        """Generates a deterministic RFC 4122 UUID from a chunk_id string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def close(self):
        """Closes the Weaviate client connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()
            app_logger.info("Weaviate client connection closed.")

    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]], namespace: str = "default"):
        """Upserts embedded document chunks into Weaviate."""
        if not documents:
            return

        if len(documents) != len(embeddings):
            raise ValueError("Document and Embedding count mismatch.")

        collection = self.client.collections.get(self.collection_name)
        objects = []
        
        for doc, emb in zip(documents, embeddings):
            properties = {
                "paper_title": doc.metadata.get("paper_title", "Unknown Title"),
                "page_number": int(doc.metadata.get("page_number", 0)),
                "chunk_id": str(doc.metadata.get("chunk_id", "")),
                "author": doc.metadata.get("author", "Unknown Author"),
                "source": doc.metadata.get("source", "unknown"),
                "embedding_model": self.model_type,
                "namespace": namespace or "default",
                "text": doc.page_content
            }
            obj_uuid = self._generate_uuid(properties["chunk_id"])
            
            objects.append(DataObject(
                properties=properties,
                vector=emb,
                uuid=obj_uuid
            ))

        app_logger.info(f"Upserting {len(objects)} objects into Weaviate collection '{self.collection_name}'...")
        
        try:
            response = collection.data.insert_many(objects)
            if response.has_errors:
                app_logger.error(f"Errors occurred during Weaviate upsert: {response.errors}")
                raise WeaviateAPIError(f"Weaviate upsert contains errors: {list(response.errors.values())[0]}")
            app_logger.info("Weaviate upsert complete.")
        except Exception as e:
            app_logger.error(f"Weaviate batch insertion failed: {str(e)}")
            raise WeaviateAPIError(f"Weaviate insertion error: {str(e)}")

    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def delete_by_ids(self, ids: List[str], namespace: str = "default"):
        """Deletes documents matching list of chunk IDs."""
        collection = self.client.collections.get(self.collection_name)
        try:
            for chunk_id in ids:
                obj_uuid = self._generate_uuid(chunk_id)
                collection.data.delete_by_id(obj_uuid)
            app_logger.info(f"Deleted {len(ids)} documents from Weaviate.")
        except Exception as e:
            app_logger.error(f"Weaviate deletion failed: {str(e)}")
            raise WeaviateAPIError(f"Weaviate delete failed: {str(e)}")

    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def delete_by_filter(self, filter_dict: Dict[str, Any], namespace: str = "default"):
        """Deletes vectors matching metadata filters."""
        collection = self.client.collections.get(self.collection_name)
        try:
            # Construct Filter object
            weaviate_filter = Filter.by_property("namespace").equal(namespace)
            for k, v in filter_dict.items():
                weaviate_filter = weaviate_filter & Filter.by_property(k).equal(v)
                
            app_logger.info(f"Deleting Weaviate documents using filter metrics...")
            collection.data.delete_many(where=weaviate_filter)
        except Exception as e:
            app_logger.error(f"Weaviate delete by filter failed: {str(e)}")
            raise WeaviateAPIError(f"Weaviate delete filter failed: {str(e)}")

    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def update_document(self, chunk_id: str, values: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None, namespace: str = "default"):
        """Updates metadata properties or vector embeddings for an existing document."""
        collection = self.client.collections.get(self.collection_name)
        try:
            obj_uuid = self._generate_uuid(chunk_id)
            kwargs = {"uuid": obj_uuid}
            if metadata:
                kwargs["properties"] = metadata
            if values:
                kwargs["vector"] = values
                
            collection.data.update(**kwargs)
            app_logger.info(f"Updated document properties for UUID '{obj_uuid}'")
        except Exception as e:
            app_logger.error(f"Weaviate update failed: {str(e)}")
            raise WeaviateAPIError(f"Weaviate update failed: {str(e)}")

    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def vector_search(self, query_vector: List[float], top_k: int = 10, namespace: str = "default", filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Queries Weaviate for vectors nearest to query_vector."""
        collection = self.client.collections.get(self.collection_name)
        try:
            # Construct filters (namespace matching + custom filter properties)
            weaviate_filter = Filter.by_property("namespace").equal(namespace)
            if filter_dict:
                for k, v in filter_dict.items():
                    weaviate_filter = weaviate_filter & Filter.by_property(k).equal(v)
                    
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=weaviate_filter,
                return_metadata=MetadataQuery(distance=True)
            )
            
            matches = []
            for obj in response.objects:
                # Weaviate distance defaults to cosine distance; similarity = 1 - distance
                similarity_score = 1.0 - (obj.metadata.distance or 0.0)
                matches.append({
                    "id": obj.properties.get("chunk_id"),
                    "score": float(similarity_score),
                    "metadata": {
                        "paper_title": obj.properties.get("paper_title"),
                        "page_number": int(obj.properties.get("page_number", 0)),
                        "chunk_id": obj.properties.get("chunk_id"),
                        "author": obj.properties.get("author"),
                        "source": obj.properties.get("source"),
                        "embedding_model": obj.properties.get("embedding_model"),
                        "text": obj.properties.get("text")
                    }
                })
            return matches
        except Exception as e:
            app_logger.error(f"Weaviate vector query failed: {str(e)}")
            raise WeaviateAPIError(f"Weaviate search failed: {str(e)}")

    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def get_ingested_papers(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Queries Weaviate for all distinct paper titles and source files."""
        collection = self.client.collections.get(self.collection_name)
        try:
            response = collection.query.fetch_objects(
                limit=1000,
                filters=Filter.by_property("namespace").equal(namespace)
            )
            
            seen = []
            papers = []
            for obj in response.objects:
                title = obj.properties.get("paper_title")
                source = obj.properties.get("source")
                author = obj.properties.get("author")
                
                if source and source not in seen:
                    seen.append(source)
                    # Count chunks for this paper
                    chunk_count = sum(1 for o in response.objects if o.properties.get("source") == source)
                    papers.append({
                        "title": title or "Unknown Paper",
                        "author": author or "Unknown Author",
                        "source": source,
                        "upload_date": "Pre-ingested",
                        "chunk_count": chunk_count
                    })
            return papers
        except Exception as e:
            app_logger.warning(f"Failed to fetch ingested papers from Weaviate: {e}")
            return []

