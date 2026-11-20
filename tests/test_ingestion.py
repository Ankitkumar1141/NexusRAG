import pytest
from src.ingestion import clean_text, guess_section

def test_clean_text_dehyphenation():
    raw = "This is a multi-\nline paragraph with broken words."
    cleaned = clean_text(raw)
    assert "multiline" in cleaned
    assert "multi-\nline" not in cleaned

def test_clean_text_whitespace_normalization():
    raw = "Too   many \t  spaces   and tabs."
    cleaned = clean_text(raw)
    assert cleaned == "Too many spaces and tabs."

def test_clean_text_newline_collapse():
    raw = "Line 1\n\n\n\n\nLine 2"
    cleaned = clean_text(raw)
    assert cleaned == "Line 1\n\nLine 2"

def test_guess_section_numbered_title():
    text = "2.1 Access Control and Security\nEngineers receive read-only access by default."
    section = guess_section(text)
    assert "Access Control and Security" in section

def test_guess_section_uppercase_header():
    text = "ANNUAL LEAVE POLICY\nEvery full-time employee receives 24 days."
    section = guess_section(text)
    assert section == "ANNUAL LEAVE POLICY"

def test_guess_section_ignores_full_sentence():
    text = "This is a full sentence that ends with a period.\nAnother sentence."
    section = guess_section(text)
    assert section == ""

