from create_VDB import QAChromaDB
from pdf_processor import PDFProcessor
from generate_answer import AnswerGenerator
from config import load_config

QA_TEMPLATE = """
You are a research assistant.

Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
if __name__ == "__main__":
    config = load_config()

    pdf_processor = PDFProcessor()

    vector_db = QAChromaDB(
        embedding_model_name=config["embedding"]["model_name"],
        collection_name=config["chroma"]["collection_name"],
        persist_directory=config["chroma"]["persist_directory"]
    )

    answer_generator = AnswerGenerator(
        template=QA_TEMPLATE,
        model_name=config["llm"]["model_name"],
        temperature=config["llm"]["temperature"]
    )

    pdf_processor.process_folder(pdf_folder=config["pdf_processor"]["pdf_path"],output_folder=config["pdf_processor"]["output_folder"])

    vector_db.ingest_files(config["pdf_processor"]["output_folder"],config["chunking"]["chunk_size"],config["chunking"]["overlap"])

    query = input("Question:")

    contexts = vector_db.query_chroma(
        query_text=query,
        n_results=config["retrieval"]["top_k"]
    )

    response = answer_generator.generate_answer(
        query=query,
        contexts=contexts
    )

    print(response.content)