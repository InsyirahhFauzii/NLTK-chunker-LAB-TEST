import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import io

# Download necessary NLTK data (only once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data()

# ─────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────
st.title("PDF Text Extraction & Sentence Chunking")
st.write("Upload a PDF file to extract text and see sentence-level chunking")

# Step 1: File uploader for PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Step 1: Read PDF using PdfReader
        pdf_reader = PdfReader(uploaded_file)
        
        # Step 2: Extract all text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""  # handle possible None
            full_text += page_text + "\n"
        
        # Clean up excessive newlines (optional)
        full_text = "\n".join(line.strip() for line in full_text.splitlines() if line.strip())
        
        st.success("PDF text extracted successfully!")
        
        # Show basic statistics
        st.write(f"**Total characters extracted**: {len(full_text):,}")
        st.write(f"**Approximate number of lines**: {len(full_text.splitlines()):,}")
        
        # Step 3: Split text into sentences using NLTK
        sentences = sent_tokenize(full_text)
        
        st.subheader("Step 3: Sample of extracted text (sentences 58–68)")
        
        if len(sentences) < 58:
            st.warning(f"The document has only {len(sentences)} sentences.")
            st.text_area("Full extracted text", full_text, height=250)
        else:
            # Show sentences 58 to 68 (0-based index → 57:68)
            sample_sentences = sentences[57:68]  # 58th to 68th sentence (inclusive)
            
            st.write("**Showing sentences 58 to 68** (1-based indexing):")
            for i, sent in enumerate(sample_sentences, start=58):
                st.markdown(f"**{i}.** {sent.strip()}")
            
            # Optional: show context around the sample
            with st.expander("Show some sentences before & after the sample"):
                start = max(0, 57-5)
                end = min(len(sentences), 68+5)
                for i in range(start, end):
                    if 58 <= i+1 <= 68:
                        st.markdown(f"**{i+1}.** {sentences[i]}")
                    else:
                        st.markdown(f"{i+1}. {sentences[i]}")
        
        # Step 4: Semantic sentence chunking (in this case: using NLTK sentence tokenizer)
        st.subheader("Step 4: Semantic Sentence Chunking (NLTK sent_tokenize)")
        st.info("The text has been split into the following semantic units (sentences):")
        
        # Show first few and last few sentences + stats
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 5 sentences**")
            for i, sent in enumerate(sentences[:5], 1):
                st.markdown(f"{i}. {sent}")
        
        with col2:
            st.write("**Last 5 sentences**")
            for i, sent in enumerate(sentences[-5:], len(sentences)-4):
                st.markdown(f"{i}. {sent}")
        
        st.write(f"**Total number of detected sentences**: {len(sentences):,}")
        
        # Optional: allow user to see more sentences if needed
        if st.button("Show first 30 sentences"):
            for i, sent in enumerate(sentences[:30], 1):
                st.markdown(f"{i}. {sent}")
                
    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
        st.info("Possible causes:\n• The PDF is scanned/image-based (no extractable text)\n• File is corrupted\n• Password protected")
else:
    st.info("Please upload a PDF file to begin extraction and sentence chunking.")