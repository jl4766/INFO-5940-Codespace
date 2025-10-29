import streamlit as st
import os
import tempfile
import shutil
from typing import List
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

#Initialize the API key
api_key = os.environ.get("API_KEY", "")

#Initialize OpenAI clients
client = OpenAI(
    api_key=api_key,
    base_url="https://api.ai.it.cornell.edu",
)

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.3,
    api_key=api_key,
    base_url="https://api.ai.it.cornell.edu"
)

#Webpage configuration
st.set_page_config(
    page_title="RAG Document Chat",
    layout="wide"
)

#Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None


##FUNCTION DEFINITIONS
#Function for load document with multiple encoding fallback plans
def load_document(file_path: str, file_type: str) -> List[Document]:
    try:
        if file_type == "txt": #txt files
            #Handles text files and tries multiple encoding formats in case one fails
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1'] #different encodings options
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]
                except UnicodeDecodeError:
                    continue
            #Final fallback read using utf-8, but ignores invalid characters
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]
            
        elif file_type == "pdf": #pdf files
            loader = PyPDFLoader(file_path) #uses a PyPDFLoader to extract the text content
            return loader.load()
        else:
            return [] #empty list if the file types are not txt or pdf
    
    #Error message        
    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
        return []

#Function for process uploaded documents and create vector store
def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int):
    if not uploaded_files: #if nothing has been uploaded by user
        st.warning("Please upload at least one document")
        return
    
    with st.spinner("Processing documents..."): #a progress spinner
        try:
            #Create temporary directory
            if st.session_state.temp_dir is None or not os.path.exists(st.session_state.temp_dir):
                st.session_state.temp_dir = tempfile.mkdtemp()
            temp_dir = st.session_state.temp_dir
            all_documents = []
            file_names = []
            
            #Process each file uploaded
            for uploaded_file in uploaded_files:
                try:
                    #Sanitize the filename
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('_', '-', '.'))
                    if not safe_filename:
                        safe_filename = f"file.{file_extension}"
                    temp_file_path = os.path.join(temp_dir, safe_filename)
                    
                    #Save the file
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    #Load document
                    docs = load_document(temp_file_path, file_extension)
                    
                    if docs:
                        #Add metadata with original filename
                        for doc in docs:
                            doc.metadata["source"] = uploaded_file.name
                        all_documents.extend(docs)
                        file_names.append(uploaded_file.name)
                
                #Error message        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            if not all_documents:
                st.error("No documents were successfully loaded")
                return
            
            #Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = text_splitter.split_documents(all_documents)
            
            #Create embeddings and vector store
            embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-large", api_key=api_key, base_url="https://api.ai.it.cornell.edu")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="document_collection")
            
            #Save to session state
            st.session_state.vectorstore = vectorstore
            st.session_state.uploaded_files_names = file_names
            st.success(f"✅ Successfully processed {len(file_names)} document(s)")
        
        #error message    
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

#Function for retrieve relevant chunks and generate answer
def retrieve_and_generate(question: str, k: int = 5) -> dict:
    if st.session_state.vectorstore is None:
        return {
            "answer": "Please upload and process documents first.",
            "sources": [],
            "retrieved_chunks": 0
        }
    
    try:
        #Retrieve relevant documents
        retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(question)
        
        #Format the context
        context = "\n\n".join([f"[From: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        #Create the prompt
        template = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            Question: {question}

            Context: {context}

            Answer:
        """
        prompt = PromptTemplate.from_template(template)
        
        #Generate answer
        messages = prompt.invoke({"question": question, "context": context})
        response = llm.invoke(messages)
        
        #Extract unique sources
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
        
        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_chunks": len(retrieved_docs)
        }
    
    #Error message
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return {
            "answer": f"An error occurred while generating the response.",
            "sources": [],
            "retrieved_chunks": 0
        }


## STREAMLIT UI
st.title("Welcome to RAG Document Chat")
st.markdown("← Use the sidebar for chunking or retrieval settings")

#Error message if API key not found
if not api_key:
    st.error("⚠️ API_KEY not found. Please set it using: `export API_KEY='your_key'`")
    st.stop()

#Sidebar for chunking settings and clear/reset chat
with st.sidebar:
    st.header("Settings")
    
    #Chunking parameters
    st.subheader("Document Chunking")
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50, 
                           help="Size of text chunks for processing")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10,
                              help="Overlap between consecutive chunks")
    
    #Retrieval parameters (set to a fixed value)
    k = 5
    
    #Display loaded documents
    if st.session_state.uploaded_files_names:
        st.subheader("Loaded Documents")
        for filename in st.session_state.uploaded_files_names:
            st.text(f"✓ {filename}")
    
    #Chat history buttons
    col1, col2 = st.columns(2)
    with col1: # Clear chat button that ONLY clear the chat history not removing any files
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2: # Reset all button that removes EVERYTHING in history
        if st.button("Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.uploaded_files_names = []
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
            st.rerun()

#Main page
uploaded_files = st.file_uploader(
    "Choose files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    help="Upload one or more .txt or .pdf files"
)

#Process Documents button
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Process Documents", disabled=not uploaded_files, type="primary", use_container_width=True):
        process_documents(uploaded_files, chunk_size, chunk_overlap)

#Display the loaded documents
if st.session_state.uploaded_files_names:
    #st.success(f"✅ {len(st.session_state.uploaded_files_names)} document(s) loaded and ready")
    with st.expander("View Loaded Documents"):
        for filename in st.session_state.uploaded_files_names:
            st.text(f"✓ {filename}")

#Status message to remind the user if nothing has been uploaded yet
if not st.session_state.vectorstore:
    st.info("💡 Upload and process documents above to start chatting")

st.divider()

#Display chatting history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                st.caption(f"Answer based on {msg.get('retrieved_chunks', 0)} relevant chunks from:")
                for source in msg["sources"]:
                    st.write(f"• {source}")

#Chat input box
if question := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.vectorstore):
    #Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    #Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = retrieve_and_generate(question, k=k)
            st.write(result["answer"])
            
            #Show document sources
            if result["sources"]:
                with st.expander("View Sources"):
                    st.caption(f"Answer based on {result.get('retrieved_chunks', 0)} relevant chunks from:")
                    for source in result["sources"]:
                        st.write(f"• {source}")
    
    #Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "retrieved_chunks": result.get("retrieved_chunks", 0)
    })