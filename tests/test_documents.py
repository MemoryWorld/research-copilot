from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from research_copilot.config import Settings
from research_copilot.documents import parse_document


def pdf_bytes(text="A public synthetic source. The budget is 42 units."):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_actual_text_pdf_extraction_has_page_provenance():
    document, chunks = parse_document("source.pdf", pdf_bytes(), Settings())
    assert document["pages"] == 1
    assert chunks[0]["page"] == 1
    assert "42 units" in chunks[0]["text"]


def test_markdown_heading_paragraph_and_overlap():
    text = "# 标题\n\n## 检索方案\n" + "中文资料" * 300
    _, chunks = parse_document("../outside.md", text.encode(), Settings())
    assert chunks[0]["document_name"] == "outside.md"
    assert chunks[0]["section"] == "检索方案"
    assert chunks[0]["page"] is None
    assert chunks[0]["paragraph"] == chunks[1]["paragraph"]
    assert chunks[0]["text"][-80:] == chunks[1]["text"][:80]


@pytest.mark.parametrize("filename,data", [("x.html", b"bad"), ("x.txt", b"\xff"), ("x.txt", b""), ("x.pdf", b"not a pdf")])
def test_invalid_document_rejected(filename, data):
    with pytest.raises(ValueError):
        parse_document(filename, data, Settings())


def test_blank_pdf_and_limits_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=50, height=50)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match="OCR"):
        parse_document("scan.pdf", stream.getvalue(), Settings())
    with pytest.raises(ValueError):
        parse_document("big.txt", b"12345", Settings(max_upload_bytes=4))
    with pytest.raises(ValueError):
        parse_document("big.txt", b"12345", Settings(max_document_chars=4))


def test_encrypted_pdf_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=50, height=50)
    writer.encrypt("private")
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match="加密"):
        parse_document("encrypted.pdf", stream.getvalue(), Settings())
