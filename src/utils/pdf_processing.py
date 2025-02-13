import fitz  # PyMuPDF for PDF processing
import streamlit as st

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") or "" for page in doc)  # Handle empty pages safely
        doc.close()
        return text.strip()  # Remove leading/trailing whitespace
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {e}")
        return ""

def remove_headers_footers(text):
    """Removes headers and footers by filtering out frequently repeated lines."""
    lines = text.split("\n")
    
    # Count occurrences of each line
    line_counts = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1

    # Remove lines that appear frequently (possible headers/footers)
    threshold = len(lines) * 0.1  # Adjust threshold as needed
    cleaned_lines = [line for line in lines if line_counts[line] <= threshold and len(line.split()) > 5]

    return " ".join(cleaned_lines)
