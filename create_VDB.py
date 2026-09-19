import os
import logging
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

class QAChromaDB:
    def __init__(self,embedding_model_name,collection_name,persist_directory="VectorDatabase/"):
        logging.basicConfig(filename='processing_log.log', level=logging.INFO)

        self.persist_directory = persist_directory

        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self.vectordb = Chroma(
            collection_name= collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function
        )

    def read_markdown_files(self, directory):
        texts = {}
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                with open(os.path.join(directory,filename),'r',encoding='utf-8') as file:
                    texts[filename] = file.read()
        return texts

    def split_text_into_chunks_with_overlap(self, text, chunk_size=512, overlap=100):
        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")

        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap phải thỏa 0 <= overlap < chunk_size")
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def store_embeddings_in_chroma(self, text_chunks, filename):
        self.vectordb.add_texts(
            texts=text_chunks,
            metadatas=[{"filename": filename, "chunk_id": i} for i in range(len(text_chunks))],
            ids = [f"{filename}_{i}" for i in range(len(text_chunks))]
        )
        print(f"Stored {len(text_chunks)} chunks for file: {filename}")

    def get_processed_files(self):
        if os.path.exists('processed_files.log'):
            with open('processed_files.log','r') as file:
                return set(file.read().splitlines())
        return set()

    def log_processed_file(self, filename):
        with open('processed_files.log', 'a') as file:
            file.write(f"{filename}\n")

    def ingest_files(self, directory,chunk_size,overlap, reset=False):

        texts = self.read_markdown_files(directory)
        processed_files = self.get_processed_files()

        for filename, text in texts.items():
            if filename in processed_files:
                logging.info(f"Skipping already processed file: {filename}")
                continue

            logging.info(f"Processing file: {filename}")
            try:
                text_chunks = self.split_text_into_chunks_with_overlap(text,chunk_size=chunk_size,overlap=overlap)
                self.store_embeddings_in_chroma(text_chunks, filename)
                self.log_processed_file(filename)
                logging.info(f"Successfully processed file: {filename}")
            except Exception:
                logging.exception(f"Error processing file: {filename}")
                raise

    def query_chroma(self, query_text, n_results=3):
        results = self.vectordb.similarity_search_with_score(query=query_text, k=n_results)
        return results

    def main(self, mode, directory=None,chunk_size = None ,overlap = None, query_text=None, reset=False, n_results=5):
        if mode == "ingest" and directory:
            print(f"Ingesting files from directory: {directory}")
            self.ingest_files(directory,chunk_size,overlap, reset=reset)
            print("Ingestion complete.")
            result = ' '
        elif mode == "query" and query_text:
            # print(f"Querying vector DB with text: '{query_text}'")
            result = self.query_chroma(query_text, n_results=n_results)
            # for document in result:
            #     print(document)
        else:
            print("Invalid mode or missing arguments. Use 'ingest' with a directory or 'query' with a query text.")
            result = " "
        return result

# Example usage:
if __name__ == "__main__":
    db = QAChromaDB('BAAI/bge-m3','QApaper')

    # # Example of data ingestion with VDB reset
    db.main(mode="ingest", directory="dataset/data_clean/textbooks/en/", reset=True)

    # Example of querying
    aa=db.main(mode="query", query_text="symptoms of drug diabetes?",n_results=1)
    print(aa)