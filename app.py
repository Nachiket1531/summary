import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import tiktoken


# 📄 Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- NEW: Token counting and chunking ---
def count_tokens(text, model="gpt-4o-mini"):
    if tiktoken is None:
        return len(text.split())  # fallback: rough word count
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def split_text_by_token_limit(text, max_tokens=100000, model="gpt-4o-mini"):
    if tiktoken is None:
        # fallback: split by words
        words = text.split()
        chunk_size = max_tokens
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start = end
    return chunks

# 🤖 Summarize using OpenAI (streamed, single chunk)
def summarize_text_with_ai(api_key, text):
    client = OpenAI(api_key=api_key)

    prompt = f"""
    You are a financial analyst AI assistant.

    Read the following financial document and provide a clear, concise summary in well-structured bullet points.

    Instructions:
    1. Extract and summarize key financial insights, decisions, and events.
    2. Clearly list all payment transactions with:
       - Amounts (in numbers)
       - Dates (if available)
       - Payees and payers (if mentioned)
       - Purpose or description of payment
    3. Highlight important financial metrics such as profits, losses, debts, loans, liabilities, or revenue figures.
    4. Keep the language professional, direct, and readable for non-financial users too.
    5. Only include information that is clearly mentioned in the document — avoid assumptions.

    Text:
    {text}
    """

    response_stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        stream=True,
        max_tokens=4096,
    )

    full_response = ""
    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content

# --- NEW: Iterative summarization for large PDFs ---
def iterative_summarize(api_key, all_text, max_tokens=100000, model="gpt-4o-mini"):
    chunks = split_text_by_token_limit(all_text, max_tokens=max_tokens, model=model)
    summary = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            input_text = chunk
            input_tokens = count_tokens(input_text, model=model)
        else:
            # Combine previous summary and next chunk, but ensure it fits in token limit
            combined = summary + "\n" + chunk
            # If too long, trim chunk
            while count_tokens(combined, model=model) > max_tokens:
                # Remove last 10% of chunk
                chunk_words = chunk.split()
                chunk = " ".join(chunk_words[:int(len(chunk_words)*0.9)])
                combined = summary + "\n" + chunk
            input_text = combined
            input_tokens = count_tokens(input_text, model=model)
        # Stream summary for this chunk
        chunk_summary = ""
        yield f"\n--- Processing chunk {i+1}/{len(chunks)} (input tokens: {input_tokens}) ---\n"
        for part in summarize_text_with_ai(api_key, input_text):
            chunk_summary += part
            yield part
        summary = chunk_summary  # Use this as the summary for next round
    yield f"\n--- Final summary complete! ---\n"
    yield summary

# 🚀 Streamlit UI
st.set_page_config(page_title="Finance PDF Summarizer", layout="centered")
st.title("📊 AI Finance PDF Summarizer")
st.markdown("Upload a financial PDF file and provide your OpenAI API key to generate a smart summary with key payment details.")

# 🔐 API Key Input
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# 📤 Upload File
uploaded_file = st.file_uploader("📄 Upload your finance PDF", type=["pdf"])

# 🧠 Run Summary
if uploaded_file and api_key:
    if st.button("🧠 Generate Summary"):
        with st.spinner("Reading and summarizing the document..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                total_tokens = count_tokens(pdf_text, model="gpt-4o-mini")
                st.info(f"Total tokens in extracted PDF text: {total_tokens}")
                if not pdf_text.strip():
                    st.error("❌ No readable text found in the PDF.")
                else:
                    st.success("✅ Summary generating...")
                    st.markdown("### 📌 Summary:")
                    placeholder = st.empty()
                    # --- Use iterative summarization, show only final summary ---
                    outputs = list(iterative_summarize(api_key, pdf_text, max_tokens=100000, model="gpt-4o-mini"))
                    if outputs:
                        final_summary = outputs[-1]
                        placeholder.markdown(final_summary)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
elif uploaded_file and not api_key:
    st.warning("⚠️ Please enter your OpenAI API key.")
