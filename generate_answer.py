from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


class AnswerGenerator:
    def __init__(self, template, model_name):
        self.llm =  ChatGroq(
            model_name = model_name,
            temperature = 0.3,
            max_retries = 2
        )
        self.prompt = PromptTemplate(
            template = template,
            input_variables = ["context", "question"]
        )
    def generate_answer(self, query, contexts):
        context = ""
        for i,(doc,score) in enumerate(contexts):
            context += f"""
[Source {i + 1}]
File: {doc.metadata.get("filename")}
Chunk: {doc.metadata.get("chunk_id")}
content: {doc.page_content}
"""
        prompt = self.prompt.format(context = context,question =query)

        response = self.llm.invoke(prompt)

        return response

