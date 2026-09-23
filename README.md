# AI Email Assistant (Version 0)

An AI-powered email assistant built with Python, Streamlit, Hugging Face Transformers, and Large Language Models. This project demonstrates the core functionality of an AI email assistant by summarizing emails, classifying their category, suggesting appropriate actions, and generating professional replies.

> **Note:** This is **Version 0 (Prototype)**. The current implementation focuses on core AI functionality using local Hugging Face models. Future versions will include a FastAPI backend, Gmail integration, Chrome extension support, and multiple AI providers.

## Features

* Email preprocessing and cleaning
* AI-generated notification summary
* Email category classification
* Suggested recipient actions
* AI-generated professional email replies
* Streamlit-based user interface

## Technology Stack

**Language**

* Python

**AI Models**

* Qwen 2.5 1.5B Instruct
* Qwen 2.5 3B Instruct
* BART Large MNLI

**Libraries & Frameworks**

* Hugging Face Transformers
* PyTorch
* Streamlit
* Regular Expressions (`re`)
* HTML Parser (`html`)
* `pathlib`

## Current Workflow

```text
Paste Email
      │
      ▼
Preprocess Email
      │
      ▼
Generate Notification
      │
      ▼
Classify Email
      │
      ▼
Suggest Available Actions
      │
      ▼
Generate AI Reply
```

## Current Limitations (Version 0)

* Uses local Hugging Face models for inference.
* No FastAPI backend.
* No Gmail integration.
* No conversation memory.
* No vector database or RAG.
* No browser extension.
* Optimized for demonstrating functionality rather than production deployment.

## Planned Roadmap

### Version 1

* Improved UI/UX
* Faster inference
* Better prompt engineering

### Version 2

* FastAPI backend
* REST API endpoints

### Version 3

* Gmail API integration
* Read and reply to emails directly

### Version 4

* Chrome Extension
* One-click email assistance

### Version 5

* Multiple AI providers (OpenAI, Claude, Gemini, Groq, Local Models)
* Provider selection

### Future Versions

* Retrieval-Augmented Generation (RAG)
* Conversation memory
* Docker deployment
* Authentication
* Production-ready architecture

## Author

**Tushar Panging**

GitHub: https://github.com/Panwithaging
