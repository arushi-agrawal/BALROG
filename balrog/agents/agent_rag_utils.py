import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

class NethackWikiSearch:
    """Handles parsing, indexing, and searching MediaWiki XML dumps with FAISS."""
    
    def __init__(self, config):
        self.model = SentenceTransformer(config.agent.embedding_model)
        self.faiss_index_path = config.agent.nethack_wiki_index
        self.storage_path = config.agent.nethack_wiki_store
        self.index = None
        self.doc_store = None
        self.top_k = config.agent.top_k

    def load_index(self):
        """Loads the FAISS index and document store if they exist."""
        if not (os.path.exists(self.faiss_index_path) and os.path.exists(self.storage_path)):
            print("No saved index found. Building the index.")

        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.storage_path, "r", encoding="utf-8") as f:
            self.doc_store = json.load(f)
            self.doc_store.pop("_global_counts", None) 

    def search(self, query):
        """Search FAISS index for similar documents and return titles + content."""
        if self.index is None or self.doc_store is None:
            print("Index not loaded. Load or build it first.")
            return []
        
        query_embedding = self.model.encode([query]).astype(np.float32)
        __build_class__, indices = self.index.search(query_embedding, self.top_k)

        retrieved_texts = [self.doc_store[list(self.doc_store.keys())[idx]]['raw_text'] for idx in indices[0]]
        return retrieved_texts
    