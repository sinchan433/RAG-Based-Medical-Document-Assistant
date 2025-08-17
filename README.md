
# 🩺 MediBot – AI Healthcare Chatbot

MediBot is an **AI-powered healthcare chatbot** designed to make medical documents interactive and easy to query. Instead of reading through long and complex clinical reports, patient records, or research articles, users can simply upload a **PDF** and ask questions in natural language. The system then provides **fast, context-aware, and reliable answers** directly from the document.

This project is built on a **Retrieval-Augmented Generation (RAG)** pipeline, combining the strengths of semantic search and large language models:

*  **Document Parsing & Chunking** → Uploaded PDFs are automatically split into smaller, meaningful text chunks for efficient storage and retrieval.
*  **Semantic Search** → Each chunk is converted into vector embeddings using HuggingFace models (`all-MiniLM-L12-v2`) and stored in **ChromaDB**, a vector database optimized for similarity search.
*  **Answer Generation** → When a query is submitted, relevant chunks are retrieved and passed to **Groq LLM**, which generates precise and context-rich responses.
*  **User Interface** → A clean, interactive **Streamlit** frontend allows users to upload files, type queries, and receive results in real time.

By integrating **LangChain**, **Groq LLM**, **HuggingFace embeddings**, and **ChromaDB**, MediBot demonstrates how modern NLP and AI technologies can be used to solve practical problems in healthcare. It helps:

*  **Doctors & Professionals** quickly extract insights from medical reports.
*  **Researchers** analyze clinical studies and medical literature more effectively.
*  **Students** interact with textbooks and study material in a Q\&A style.

Beyond healthcare, the same architecture can be adapted to other domains such as **law, education, or finance**, making it a flexible and scalable solution for document-based knowledge retrieval.

MediBot is a demonstration of how **Large Language Models (LLMs)** and **vector databases** can be combined to bridge the gap between raw text documents and actionable knowledge, showing the power of **applied NLP and AI engineering**.

