# 🎯 Interview Q&A Generator

AI-powered tool that generates interview questions and answers from PDF documents using OpenAI GPT-3.5 and LangChain.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-red.svg)

## ✨ Features

- 📄 **PDF Upload** - Drag & drop interface with real-time validation
- 🤖 **AI Question Generation** - Smart interview questions using GPT-3.5
- 💡 **RAG-based Answers** - Accurate answers using document retrieval
- 📥 **CSV Export** - Download Q&A pairs instantly
- ⚡ **UI** - Responsive design with loading animations

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Shakiththiyanofficial/interview-qa-generator.git
cd interview-qa-generator
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Application
```bash
python app.py
```
Visit: `http://localhost:8000`

## 📋 Requirements

```txt
fastapi==0.116.1
uvicorn==0.35.0
python-multipart==0.0.20
aiofiles==24.1.0
jinja2==3.1.6
langchain==0.3.27
langchain-community==0.3.27
langchain-core==0.3.74
langchain-text-splitters==0.3.9
openai==1.99.9
pypdf==6.0.0
PyPDF2==3.0.1
faiss-cpu==1.11.0.post1
python-dotenv==1.1.1
tiktoken==0.11.0
pydantic==2.11.7
requests==2.32.4
```

## 📁 Project Structure

```
interview-qa-generator/
├── app.py              # FastAPI application
├── src/
│   ├── helper.py       # Core AI processing
│   └── prompt.py       # Question templates
├── templates/
│   └── index.html      # Web interface
└── static/             # File storage (auto-created)
```

## 🔧 Usage

1. **Upload PDF** → Drag & drop your coding document
2. **Generate Q&A** → AI processes and creates questions
3. **Download CSV** → Get your interview questions & answers

## 🤖 Technology Stack

- **FastAPI** - Web framework
- **LangChain** - Document processing & AI chains
- **OpenAI GPT-3.5** - Question & answer generation
- **FAISS** - Vector similarity search
- **Modern HTML/CSS/JS** - Responsive UI

## 📊 Sample Output

```csv
Question,Answer
"What is the time complexity of binary search?","Binary search has O(log n) time complexity..."
"How do you implement a stack data structure?","A stack can be implemented using arrays or linked lists..."
```

## 🐛 Common Issues

- **API Key Error**: Ensure `.env` file contains valid `OPENAI_API_KEY`
- **Module Not Found**: Create empty `src/__init__.py` file
- **Port Busy**: Change port in `app.py` from 8000 to another port

## 🚀 Future Features

- Multiple file formats (Word, PowerPoint)
- Question difficulty selection
- Custom prompt templates
- Bulk file processing
