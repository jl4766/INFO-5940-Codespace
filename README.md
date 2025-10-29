# INFO 5940 Assignment 1
Welcome to my INFO 5940 Assignment 1 codespace for RAG Document Chat. Please see the instructions below for specific steps for using this RAG.

## Features
1. Upload documents
2. Customizable chunking strategies on the left sidebar
3. Multi-turn conversational interface with chat history
   * All your questions and answers are saved in the current session
   * Scroll up to review previous conversations
   * The AI maintains context across multiple questions
4. Source attribution showing which documents were used
5. Clear chat or reset all functionality
   * Clear Chat: Removes conversation history but keeps your documents loaded
      * Use this when you want to start a fresh conversation with the same documents
   * Reset All: Completely removes everything
      * Clears conversation history
      * Removes all uploaded documents
      * Deletes the vector database
      * Use this to start completely fresh with new documents
6. Supported File Types
   * .txt files (text documents)
   * .pdf files (PDF documents)
7. File Encoding
   * The system automatically handles multiple text encodings: UTF-8, Latin-1, CP1252, ISO-8859-1
8. Source Attribution
   * Every answer includes a "View Sources" section
   * Shows which documents contributed to the answer
   * Provides transparency about where information came from

### Getting Started 
## Step 1: Set up API Key and run streamlit:
To run the Streamlit app and set up the key at the same time, run the commands below in terminal:
   ```bash
   API_KEY="your_actual_API_KEY" streamlit run chat_with_pdf.py
   ```

## Step 2: (Optional) Changing chunking or retrieval settings:
1. Located on the left sidebar, you can adjust the chunk size, chunk overlap, and number of retrieval chunks to your needs
   * Chunk Size: Controls how large each text chunk is (100-1000 characters)
      * Default: 500 characters
      * Larger chunks provide more context but fewer total chunks
      * Smaller chunks allow for more precise retrieval
   * Chunk Overlap: How much adjacent chunks overlap (0-200 characters)
      * Default: 50 characters
      * More overlap ensures better context continuity
      * Less overlap is more efficient but may miss connections

## Step 3: Upload documents 
1. Click the "Browse files" or drag the files in to the upload documents area/function
   (You can select one or more .txt or .pdf files)
2. When the files has successfully uploaded, click "Process Documents" to begin file processing
   * A spinner will show "Processing documents..." while the system:
      * Saves your files temporarily
      * Splits them into chunks based on your settings
      * Creates embeddings for semantic search
      * Stores them in the vector database
3. Once processing is complete, you'll see a success message with the number of documents loaded
4. You can view your loaded documents by clicking "View Loaded Documents" expander, it will also be shown in the sidebar on the left

## Step 4: Ask your questions
1. Once the file is successfully processed, type your question below in the input bar
2. The system will:
   * Search for the most relevant chunks from your documents
   * Use those chunks as context for the AI
   * Generate a concise, accurate answer based on your documents
3. Once a response is produced, you can ask more questions by typing in the input bar 
4. Click "View Sources" below each answer to see which documents were used