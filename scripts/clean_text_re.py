import os
import json
import re
import pathlib
from pathlib import Path

grandparent_dir = pathlib.Path(__file__).resolve().parent.parent

def clean_scientific_text(text):
    patterns = [
        # 1. 引用标记（支持多种格式）
        r"\\?\[\@[^\]]+\]\\?",  # [@b1] 或 \[@b1\]
        r"\^\[\@[^\]]+\]\^",  # ^[@b1]^
        r"\\?\(#[^)]+\)\\?",  # (#s1) 或 \(#s1\)

        # 2. LaTeX相关（命令、环境、引用）
        r"\\[a-z]+\{.*?\}",  # \usepackage{...}
        r"\\[^a-z]\S*",  # \* 等特殊命令
        r"\\begin\{.*?\}.*?\\end\{.*?\}",  # LaTeX环境

        # 3. 图片/表格标记
        r"!\[.*?\]\(.*?\)",  # ![caption](path)
        r"\{ref-type=\"[^\"]+\"\}",  # {ref-type="table"}
        r"\\?\[(Table|Fig|Figure)[^\]]*\]\\?",  # [Table 1]

        # 4. 链接和标识符
        r"https?://\S+",  # http/https链接
        r"\bDOI:\S+",  # DOI:xxxx
        r"\bDoi:\S+",  # Doi:xxxx
        r"<\S+>",  # <xxx@xxx>

        # 5. 特殊格式
        r"={5,}",  # ====标题下划线====
        r"\-{3,}",  # ---分割线---
        r"\#{2,}",

        # 6. 表格结构（去除表格线）
        r"\|.*\|",  # | xxx | xxx |
        r"_{3,}",  # ___表格下划线___

        # 7. 其他
        r"\*{2}(.*?)\*{2}",  # **强调文本** → 保留文本
        r"\\'",  # 转义单引号
        # 移除长串连续数字（如000..., 111...）
        r'0{10,}',  # 10个或更多连续的0
        r'1{10,}',  # 10个或更多连续的1
    ]

    # 先删除不需要的内容
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # 后处理：清理残余字符
    text = re.sub(r"\(\s*\)", "", text)  # 空括号
    text = re.sub(r"\s+([.,;:])", r"\1", text)  # 标点前空格
    text = re.sub(r"\.{3,}", "...", text)  # 统一省略号

    # 移除所有换行符，并用单个空格替换
    text = re.sub(r"[\n\r]+", " ", text)

    # 合并多余空格（保留单个空格）
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text

def process_jsonl_file(input_path, output_path):
    """处理单个jsonl文件"""
    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                data = json.loads(line.strip())
                if "text" in data:
                    cleaned_text = clean_scientific_text(data["text"])
                    outfile.write(json.dumps({"text": cleaned_text}, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print(f"警告：跳过无效的JSON行: {line.strip()}")
                continue
def process_directory(input_dir, output_dir):
    """处理目录下所有jsonl文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"cleaned_{filename}")

            print(f"正在处理: {filename}")
            process_jsonl_file(input_path, output_path)
            print(f"已完成: {filename} -> cleaned_{filename}")


if __name__ == "__main__":
    input_directory = f"{grandparent_dir}/data/raw/text"
    output_directory = f"{grandparent_dir}/data/processed/text"
    process_directory(input_directory, output_directory)
    print("所有文件处理完成！")