"""Parse uploaded bytes only. No URL fetches, local-path reads, or OCR claims."""
from dataclasses import dataclass, asdict
from io import BytesIO
import re
import uuid

from .config import Settings


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    text: str
    page: int | None
    section: str
    paragraph: int
    start: int
    end: int

    def dump(self):
        return asdict(self)


def parse_document(filename: str, data: bytes, settings: Settings) -> tuple[dict, list[dict]]:
    if not data or len(data) > settings.max_upload_bytes:
        raise ValueError("文档为空或超过 8 MB 上传限制")
    name = filename.replace("\\", "/").rsplit("/", 1)[-1][:160]
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in {"txt", "md", "pdf"}:
        raise ValueError("仅支持 UTF-8 txt、md 和含文本层的 pdf")
    pages = []
    if extension == "pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ValueError("不支持加密 PDF")
            if len(reader.pages) > settings.max_pdf_pages:
                raise ValueError("PDF 超过 80 页限制")
            total = 0
            for index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                total += len(text)
                if total > settings.max_document_chars:
                    raise ValueError("提取文本超过文档字符上限")
                pages.append((index, text))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF 无法解析，请提供有效且带文本层的 PDF") from exc
    else:
        try:
            pages = [(None, data.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("文本文件必须使用 UTF-8 编码") from exc
    if sum(len(text) for _, text in pages) > settings.max_document_chars:
        raise ValueError("提取文本超过文档字符上限")
    document_id = uuid.uuid4().hex
    chunks = []
    section = "正文"
    paragraph = 0
    for page_number, text in pages:
        for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            if lines[0].startswith("#"):
                section = lines[0].lstrip("# ")[:100] or "正文"
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            paragraph += 1
            for start in range(0, len(block), 520):
                end = min(start + 600, len(block))
                chunks.append(Chunk(uuid.uuid4().hex, document_id, name, block[start:end],
                                    page_number, section, paragraph, start, end).dump())
                if end == len(block):
                    break
                if len(chunks) > settings.max_chunks:
                    raise ValueError("切片数量超过上限，请拆分文档")
    if not chunks:
        raise ValueError("未提取到正文；扫描 PDF 需要先做 OCR，本演示不包含 OCR")
    if len(chunks) > settings.max_chunks:
        raise ValueError("切片数量超过上限，请拆分文档")
    return {"id": document_id, "name": name, "pages": len(pages), "chunks": len(chunks)}, chunks
