import streamlit as st
from PIL import Image
import pytesseract
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import PyPDF2
from PyPDF2 import PdfReader
from autocorrect import Speller
import language_tool_python
import io

# MODEL AND TOKENIZER
checkpoint = "D:/Project School/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32, token="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")
spell = Speller()

def perform_ocr(image):
    text = pytesseract.image_to_string(image)
    return text

import tempfile
import logging

def file_preprocessing(uploaded_file):
    if uploaded_file is None:
        st.error("No file uploaded!")
        return None

    if not uploaded_file:
        return None

    # Calculate file size from content
    file_content = uploaded_file.read()
    file_size = len(file_content)

    # Log file size for debugging 
    logging.info(f"Uploaded file size: {file_size}")

    try:
        # Check for empty file
        if len(file_content) == 0:
            st.error("Uploaded file is empty!")
            return None

        # Check MIME type 
        if uploaded_file.type not in ['application/pdf']:
            st.error("Invalid file format. Please upload a PDF file.")
            return None
        # Process the file using PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()
        return text

    except PyPDF2.errors.PdfReadError as e:
        # Handle potential PDF format errors
        st.error(f"Error processing PDF: {e}. The file might be corrupted.")
        logging.error(f"Error processing PDF: {e}")
        return None

    except Exception as e:  # Catch other unexpected errors
        st.error(f"Error processing file: {e}")
        logging.error(f"Error processing file: {e}")
        return None



def count_words(text):
    words = text.split()
    return len(words)

def llm_pipeline(text, summary_length_words):
    pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer)
    summary = ""
    total_words = 0
    if text is not None:  # Check for None before splitting
        sentences = text.split(".")
        for sentence in sentences:
            result = pipe_sum(sentence, max_length=summary_length_words, min_length=int(summary_length_words * 0.6), do_sample=False)
            sentence_summary = result[0]['summary_text']
            words_in_sentence = count_words(sentence_summary)
            if total_words + words_in_sentence <= summary_length_words:
                summary += sentence_summary + " "
                total_words += words_in_sentence
            if total_words >= summary_length_words:
                break
    return summary

def check_grammar_vocabulary(input_text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(input_text)
    corrected_text = tool.correct(input_text)
    return matches, corrected_text

@st.cache_data
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def process_text(input_text, summary_length_words):
    pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer)
    summary = ""
    total_words = 0
    sentences = input_text.split(".")
    for sentence in sentences:
        result = pipe_sum(sentence, max_length=summary_length_words, min_length=int(summary_length_words * 0.6), do_sample=False)
        sentence_summary = result[0]['summary_text']
        words_in_sentence = count_words(sentence_summary)
        if total_words + words_in_sentence <= summary_length_words:
            summary += sentence_summary + " "
            total_words += words_in_sentence
        if total_words >= summary_length_words:
            break
    return summary

# Streamlit code
st.set_page_config(layout='wide', page_title="Summarization")

def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        st.error("No file uploaded!")
        return None

    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def grammar_vocabulary_tab():
    input_text = st.text_area("Input Text for Grammar and Vocabulary Check")
    if st.button("Check Grammar and Vocabulary"):
        matches, corrected_text = check_grammar_vocabulary(input_text)
        if matches:
            for match in matches:
                try:
                    line_info = f" (Line {match.fromy}, Column {match.fromx})"
                except AttributeError:
                    line_info = ""
            st.subheader("Corrected Text:")
            st.markdown(f"<pre>{corrected_text}</pre>", unsafe_allow_html=True)

def main():
    selected_task = st.sidebar.selectbox("Select Task", ["Summarization", "Grammar and Vocabulary Check"])
    st.sidebar.title('Options')
    uploaded_files = st.sidebar.file_uploader("Upload your PDF files, images with text, or text files", type=['pdf', 'png', 'jpg', 'txt'], accept_multiple_files=True)
    summary_length_words = st.sidebar.slider("Select Summary Length (in words)", min_value=50, max_value=1000, value=250, step=10)

    if selected_task == "Summarization":
        st.title('Text and Image Summarization')
        input_text = st.sidebar.text_area("Input Text for Summarization")
        get_summary_button = st.button("Get Summary from Input Text")
        if get_summary_button:
            summary = process_text(input_text, summary_length_words)
            st.success(summary)

        text_from_file = ""
        with st.sidebar.expander("Question and Answer"):
            question = st.text_input("Ask a question related to the text")
            corrected_question = spell(question)

            if uploaded_files:
                file_names = [uploaded_file.name for uploaded_file in uploaded_files]
            else:
                file_names = [input_text]

            selected_file = st.selectbox("Select a file", file_names)

            if selected_file:
                selected_uploaded_file = next((file for file in uploaded_files if file.name == selected_file), None)
                if selected_uploaded_file:
                    text_from_file = extract_text_from_file(selected_uploaded_file)

                if st.button("Submit"):
                    if text_from_file and corrected_question:
                        qa_input = f"question: {corrected_question} context: {text_from_file}"
                        answer = base_model.generate(input_ids=tokenizer.encode(qa_input, return_tensors="pt"), max_length=100)
                        decoded_answer = tokenizer.decode(answer[0], skip_special_tokens=True)
                        st.write("Answer:", decoded_answer)
                    else:
                        st.warning("Please select a file")
            else:
                st.warning("Please upload a file")

    elif selected_task == "Grammar and Vocabulary Check":
        st.title('Grammar and Vocabulary Check')
        grammar_vocabulary_tab()

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == 'application/pdf':
                if st.button(f"Get Summary of {uploaded_file.name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("Uploaded file")
                        displayPDF(uploaded_file)
                    with col2:
                        st.info("Summary")
                        summary = llm_pipeline(file_preprocessing(io.BytesIO(uploaded_file.read())), summary_length_words)
                        st.success(summary)

            elif uploaded_file.type.startswith('image/'):
                if st.button(f"Get Summary of {uploaded_file.name}"):
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Uploaded image: {uploaded_file.name}", use_column_width=True)
                    text_from_image = perform_ocr(image)
                    summary = process_text(text_from_image, summary_length_words)
                    st.success(summary)

            elif uploaded_file.type == 'text/plain':
                if st.button(f"Get Summary of {uploaded_file.name}"):
                    text = uploaded_file.getvalue().decode("utf-8")
                    if text.strip():
                        summary = process_text(text, summary_length_words)
                        st.success(summary)
                    else:
                        st.warning("The uploaded text file is empty. Please upload a text file with content.")

if __name__ == '__main__':
    main()

                   