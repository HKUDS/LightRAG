## Quick start
* install textract
```bash
pip install textract
```
*example
```bash
import textract
# 指定要提取文本的文件路径
file_path = 'path/to/your/file.pdf'
# 从文件中提取文本
text_content = textract.process(file_path)
# 打印提取的文本
print(text_content.decode('utf-8'))
```


