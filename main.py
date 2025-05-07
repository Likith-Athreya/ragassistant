import os
import glob
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from numexpr import evaluate
from PyDictionary import PyDictionary

# Configuration
MODEL_NAME = "google/flan-t5-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

@st.cache_resource
def load_data():
    """Load and process documents"""
    docs = []
    for file in glob.glob("docs/*.txt"):
        loader = TextLoader(file)
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

@st.cache_resource
def create_vectorstore():
    """Create FAISS vector store"""
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(load_data(), embeddings)

@st.cache_resource
def load_llm():
    """Load language model"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def main():
    st.title("🤖 RAG Assistant")
    
    # Initialize components
    vectordb = create_vectorstore()
    llm = load_llm()
    
    # User input
    query = st.text_input("Ask a question:")
    
    if query:
        if any(kw in query.lower() for kw in ["calculate", "compute"]):
            try:
                result = evaluate(query.lower().replace("calculate", "").replace("compute", ""))
                st.success(f"🔢 Result: {result}")
            except:
                st.error("❌ Invalid calculation")
        
        elif "define" in query.lower():
            word = query.lower().split("define")[-1].strip()
            try:
                definition = PyDictionary().meaning(word)
                st.json(definition)
            except:
                st.error("❌ Definition not found")
        
        else:
            docs = vectordb.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            response = llm(f"Question: {query}\nContext: {context}")[0]['generated_text']
            st.markdown(f"**Answer:** {response}")
            
            with st.expander("View context"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Context {i}:**")
                    st.text(doc.page_content)

if __name__ == "__main__":
    main()