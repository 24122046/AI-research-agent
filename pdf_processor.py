import os
import logging
import pymupdf4llm
import hashlib
import json

class PDFProcessor:

    def __init__(self):
        logging.basicConfig(
            filename="processing_log.log",
            level=logging.INFO
        )

    def pdf_to_markdown(self, pdf_path):
        markdown = pymupdf4llm.to_markdown(pdf_path)
        return markdown


    def save_markdown(self, markdown, output_path):
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(markdown)

    def process_pdf(self, pdf_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        markdown = self.pdf_to_markdown(pdf_path)

        filename = os.path.basename(pdf_path)

        markdown_filename = (
            os.path.splitext(filename)[0] + ".md"
        )

        output_path = os.path.join(
            output_folder,
            markdown_filename
        )

        self.save_markdown(
            markdown,
            output_path
        )

        return output_path

    def process_folder(self, pdf_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        # Mỗi thư mục Markdown có trạng thái xử lý PDF riêng
        state_path = os.path.join(output_folder, "processed_pdf.json")

        processed_files = {}

        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as file:
                processed_files = json.load(file)

        for filename in os.listdir(pdf_folder):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(pdf_folder, filename)

            if not os.path.isfile(pdf_path):
                continue

            # Dùng đường dẫn đầy đủ để phân biệt PDF trùng tên
            file_key = os.path.abspath(pdf_path)

            markdown_path = os.path.join(
                output_folder,
                os.path.splitext(filename)[0] + ".md"
            )

            try:
                with open(pdf_path, "rb") as file:
                    content_hash = hashlib.sha256(file.read()).hexdigest()

                # PDF không đổi và Markdown vẫn tồn tại thì bỏ qua
                if (
                        processed_files.get(file_key) == content_hash
                        and os.path.isfile(markdown_path)
                ):
                    logging.info(f"Skipping unchanged PDF: {filename}")
                    continue

                self.process_pdf(pdf_path, output_folder)

                # Chỉ ghi nhận sau khi tạo Markdown thành công
                processed_files[file_key] = content_hash

                with open(state_path, "w", encoding="utf-8") as file:
                    json.dump(
                        processed_files,
                        file,
                        ensure_ascii=False,
                        indent=2
                    )

                logging.info(f"Successfully processed PDF: {filename}")

            except Exception:
                logging.exception(f"Error processing PDF: {filename}")
                raise