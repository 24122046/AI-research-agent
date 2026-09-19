import os
import logging
import pymupdf4llm


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


    def get_processed_files(self):
        if os.path.exists("processed_pdf.log"):
            with open(
                "processed_pdf.log",
                "r",
                encoding="utf-8"
            ) as file:
                return set(file.read().splitlines())

        return set()


    def log_processed_file(self, filename):
        with open(
            "processed_pdf.log",
            "a",
            encoding="utf-8"
        ) as file:
            file.write(f"{filename}\n")


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

        processed_files = self.get_processed_files()

        for filename in os.listdir(pdf_folder):

            if not filename.lower().endswith(".pdf"):
                continue

            markdown_path = os.path.join(
                output_folder,
                os.path.splitext(filename)[0] + ".md"
            )

            if filename in processed_files and os.path.isfile(markdown_path):

                logging.info(
                    f"Skipping already processed PDF: {filename}"
                )

                continue

            pdf_path = os.path.join(
                pdf_folder,
                filename
            )

            logging.info(
                f"Processing PDF: {filename}"
            )

            try:
                output_path = self.process_pdf(
                    pdf_path,
                    output_folder
                )

                self.log_processed_file(filename)

                logging.info(
                    f"Successfully processed PDF: "
                    f"{filename} -> {output_path}"
                )

            except Exception as e:

                logging.error(
                    f"Error processing PDF "
                    f"{filename}: {e}"
                )