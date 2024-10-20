# LlamaIndex-chatbot-with-advanced-search-and-RAG

Please refer to thie [REPOSITORY](https://github.com/sulaiman-shamasna/Advanced-Chatbot-Agentic-RAG-with-LlamaIndex) instead!
This chatbot utilizes the concept of Agentic RAG to develop an advanced chatbot that is capable of reasoning among complicated-context documents

## Project Structure

```plaintext
LlamaIndex-chatbot-with-advanced-search-and-RAG/
    ├── backend/
    │   ├── app/
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   ├── langchain_bot.py
    │   ├── Dockerfile
    ├── frontend/
    │   ├── app/
    │   │   ├── main.py
    │   ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── venv
    ├── .env
    ├── .gitignore
    ├── docs/
    ├── 
```

## Environment Setup

1. with virtual environment

2. with docker


## Access the Application

- The FastAPI backend will be available at http://localhost:8000.
- The Streamlit frontend will be available at http://localhost:8501.

## Additional Considerations (@TODO)

- Error Handling: Implement proper error handling in both the backend and frontend.
- Security: Consider securing your APIs and sanitizing user inputs.
- Logging: Add logging to capture important events and errors.
- Testing: Write tests for both backend and frontend to ensure reliability.
- Environment Variables: Use environment variables for configuration (e.g., ports, URLs).
