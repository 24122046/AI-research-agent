from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import load_config
from create_VDB import QAChromaDB
from generate_answer import AnswerGenerator
from main import QA_TEMPLATE

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://research-companion-nlp.nongtrainato.chatgpt.site"
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Khởi tạo một lần khi nạp ứng dụng
config = load_config()

vector_db = QAChromaDB(
    embedding_model_name=config["embedding"]["model_name"],
    collection_name=config["chroma"]["collection_name"],
    persist_directory=config["chroma"]["persist_directory"],
)

answer_generator = AnswerGenerator(
    template=QA_TEMPLATE,
    model_name=config["llm"]["model_name"],
    temperature=config["llm"]["temperature"],
)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Câu hỏi không được để trống.",
        )

    contexts = vector_db.query_chroma(
        query_text=question,
        n_results=config["retrieval"]["top_k"],
    )

    if not contexts:
        return {
            "answer": "Chưa tìm thấy tài liệu để trả lời.",
            "sources": [],
        }

    response = answer_generator.generate_answer(
        query=question,
        contexts=contexts,
    )

    return {
        "answer": response.content,
        "sources": [
            {
                "id": i + 1,
                "filename": doc.metadata.get("filename"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
            }
            for i, (doc, score) in enumerate(contexts)
        ],
    }