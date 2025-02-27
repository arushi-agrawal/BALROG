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
        # self.wiki_path = config.agent.nethack_wiki
        self.faiss_index_path = config.agent.nethack_wiki_index
        self.storage_path = config.agent.nethack_wiki_store
        self.index = None
        self.doc_store = None
        self.top_k = config.agent.top_k

    # def __parse_xml(self):
    #     """Parses MediaWiki XML and extracts full text per page."""
    #     ns = {"mw": "http://www.mediawiki.org/xml/export-0.10/"}
    #     tree = ET.parse(self.wiki_path)
    #     root = tree.getroot()

    #     pages = root.findall(".//mw:page", namespaces=ns)
    #     extracted = []

    #     for page in pages:
    #         title = page.find("mw:title", namespaces=ns).text
    #         revision = page.find(".//mw:revision/mw:text", namespaces=ns)

    #         if revision is not None and revision.text:
    #             text_content = revision.text.strip()
    #             extracted.append((title, text_content))

    #     return extracted
    

    # def __build_index(self):
    #     """Encodes pages with embeddings and stores in FAISS HNSW index along with full text."""
    #     data = self.__parse_xml()
    #     titles, texts = zip(*data)  # Extract titles and content
        
    #     # Convert text into embeddings
    #     embeddings = self.model.encode(texts, convert_to_numpy=True)
    #     faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    #     # Use HNSW for scalable nearest-neighbor search
    #     dim = embeddings.shape[1]
    #     self.index = faiss.IndexHNSWFlat(dim, 32)
    #     self.index.hnsw.efConstruction = 128  # Better recall
    #     self.index.add(embeddings)

    #     # Store full content alongside titles
    #     self.doc_store = [{"title": t, "content": c} for t, c in zip(titles, texts)]

    #     # Save FAISS index and document store
    #     faiss.write_index(self.index, self.faiss_index_path)
    #     with open(self.storage_path, "wb") as f:
    #         pickle.dump(self.doc_store, f)

    #     print("FAISS index and document store saved.")


    def load_index(self):
        """Loads the FAISS index and document store if they exist."""
        if not (os.path.exists(self.faiss_index_path) and os.path.exists(self.storage_path)):
            print("No saved index found. Building the index.")
            # self.__build_index()

        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.storage_path, "r", encoding="utf-8") as f:
            self.doc_store = json.load(f)
            self.doc_store.pop("_global_counts", None) 

        # with open(self.storage_path, "rb") as f:
        #     self.doc_store = pickle.load(f)

        # return print("Loaded FAISS index and document store from disk.")


    def search(self, query):
        """Search FAISS index for similar documents and return titles + content."""
        if self.index is None or self.doc_store is None:
            print("Index not loaded. Load or build it first.")
            return []

        # query_embedding = self.model.encode([query], convert_to_numpy=True)
        # faiss.normalize_L2(query_embedding)  # Normalize query

        # distances, indices = self.index.search(query_embedding, self.top_k)

        # return [(self.doc_store[idx]["title"], self.doc_store[idx]["content"]) for idx in indices[0]]
        query_embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, self.top_k)

        retrieved_texts = [self.doc_store[list(self.doc_store.keys())[idx]]['raw_text'] for idx in indices[0]]
        return retrieved_texts
    