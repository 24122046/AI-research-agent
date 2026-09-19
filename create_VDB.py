import os
import logging
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import hashlib
import json

class QAChromaDB:
    def __init__(self,embedding_model_name,collection_name,persist_directory="VectorDatabase/"):
        logging.basicConfig(filename='processing_log.log', level=logging.INFO)

        self.persist_directory = persist_directory

        self.processed_log_path = os.path.join(
            persist_directory,
            f"{collection_name}_processed.json"
        )

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

    def ingest_files(self, directory, chunk_size, overlap, reset=False):
        texts = self.read_markdown_files(directory)

        os.makedirs(self.persist_directory, exist_ok=True)

        # Đọc trạng thái: tên file -> hash nội dung
        processed_files = {}

        if os.path.isfile(self.processed_log_path):
            with open(self.processed_log_path, "r", encoding="utf-8") as file:
                processed_files = json.load(file)

        for filename, text in texts.items():
            content_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            # Tìm các chunk hiện có của tài liệu
            old_ids = self.vectordb.get(
                where={"filename": filename}
            )["ids"]

            # Chỉ bỏ qua nếu nội dung không đổi và database còn dữ liệu
            if processed_files.get(filename) == content_hash and old_ids:
                logging.info(f"Skipping unchanged file: {filename}")
                continue

            try:
                text_chunks = self.split_text_into_chunks_with_overlap(
                    text,
                    chunk_size=chunk_size,
                    overlap=overlap
                )

                # Xóa toàn bộ chunk cũ để tránh giữ lại nội dung lỗi thời
                if old_ids:
                    self.vectordb.delete(ids=old_ids)

                if text_chunks:
                    self.store_embeddings_in_chroma(text_chunks, filename)

                # Chỉ cập nhật trạng thái sau khi xử lý thành công
                processed_files[filename] = content_hash

                with open(
                        self.processed_log_path, "w", encoding="utf-8"
                ) as file:
                    json.dump(
                        processed_files,
                        file,
                        ensure_ascii=False,
                        indent=2
                    )

                logging.info(f"Successfully processed file: {filename}")

            except Exception:
                logging.exception(f"Error processing file: {filename}")
                raise

    def query_chroma(self, query_text, n_results=3):
        results = self.vectordb.similarity_search_with_score(query=query_text, k=n_results)
        return results

    def main(self, mode, directory=None,chunk_size = 512 ,overlap = 100, query_text=None, reset=False, n_results=5):
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