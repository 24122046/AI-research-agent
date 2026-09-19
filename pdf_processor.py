import os
import pymupdf4llm


class PDFProcessor:

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
        filename = os.path.splitext(filename)[0] + ".md"

        output_path = os.path.join(
            output_folder,
            filename
        )

        self.save_markdown(
            markdown=markdown,
            output_path=output_path
        )

        return output_path