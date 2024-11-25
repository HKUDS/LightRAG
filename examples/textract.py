import csv
import json
import PyPDF2
import pandas as pd
import textract
from docx import Document


def read_text_file(file_path):
    """根据文件类型读取文件内容"""
    # 获取文件扩展名
    file_extension = file_path.split('.')[-1].lower()

    # 读取普通文本文件（.txt）
    if file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # 读取CSV文件（.csv）
    elif file_extension == 'csv':
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return [row for row in reader]

    # 读取JSON文件（.json）
    elif file_extension == 'json':
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    # 读取PDF文件（.pdf）
    elif file_extension == 'pdf':
        # 尝试使用 PyPDF2 提取文本
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF with PyPDF2: {e}")
            # 如果 PyPDF2 失败，尝试使用 textract
            try:
                return textract.process(file_path).decode('utf-8')
            except Exception as e:
                print(f"Error reading PDF with textract: {e}")
                return None

    # 读取 DOCX 文件（.docx）
    elif file_extension == 'docx':
        try:
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
            return None

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


