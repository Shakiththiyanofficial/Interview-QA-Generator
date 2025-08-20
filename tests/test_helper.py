import pytest
from src.helper import file_processing, clean_and_filter_questions, validate_environment
from langchain.docstore.document import Document

# -----------------------
# TEST clean_and_filter_questions
# -----------------------

def test_clean_and_filter_questions_valid():
    text = """
    1. What is Python?
    2. Explain the concept of OOP.
    3. Why is testing important?
    """
    result = clean_and_filter_questions(text)
    assert len(result) == 3
    assert "What is Python?" in result

def test_clean_and_filter_questions_invalid():
    text = """
    1. asdfasdfasdf
    2. 12345
    """
    result = clean_and_filter_questions(text)
    assert result == []  # No valid questions

# -----------------------
# TEST validate_environment
# -----------------------

def test_validate_environment_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        validate_environment()

def test_validate_environment_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    validate_environment()  # Should not raise

# -----------------------
# TEST file_processing with mocking
# -----------------------

def test_file_processing_missing_file(tmp_path):
    fake_file = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        file_processing(str(fake_file))

def test_file_processing_empty_pdf(monkeypatch, tmp_path):
    # Mock PyPDFLoader.load to return empty list
    monkeypatch.setattr("src.helper.PyPDFLoader.load", lambda self: [])
    
    fake_file = tmp_path / "empty.pdf"
    fake_file.write_text("dummy")  # create dummy file
    with pytest.raises(ValueError, match="No content found"):
        file_processing(str(fake_file))

def test_file_processing_valid_pdf(monkeypatch, tmp_path):
    # Mock PyPDFLoader.load to return fake documents
    fake_docs = [Document(page_content="This is page one."),
                 Document(page_content="This is page two.")]
    monkeypatch.setattr("src.helper.PyPDFLoader.load", lambda self: fake_docs)

    fake_file = tmp_path / "valid.pdf"
    fake_file.write_text("dummy")  # create dummy file
    
    q_chunks, a_chunks = file_processing(str(fake_file))
    assert len(q_chunks) > 0
    assert len(a_chunks) > 0
    assert any("page one" in d.page_content.lower() for d in q_chunks)
