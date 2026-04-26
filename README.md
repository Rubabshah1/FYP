# Alkhidmat Public Chat Portal

A comprehensive RAG (Retrieval-Augmented Generation) based chat portal for Alkhidmat Foundation, featuring AI-powered responses, human agent support, and comprehensive analytics.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [GPU Acceleration (Mac)](#gpu-acceleration-mac)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Alkhidmat Public Chat Portal is a full-stack application that provides:

- **User Interface**: OTP-based authentication, chat with RAG AI, and seamless escalation to human agents
- **Agent Dashboard**: Ticket management, real-time chat with users, and ticket resolution
- **Admin Dashboard**: Comprehensive analytics, ticket monitoring, and system insights

The system uses a RAG (Retrieval-Augmented Generation) pipeline to answer user queries from a knowledge base, with automatic ticket creation when confidence scores are low.

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Frontend  │────────▶│  FastAPI     │────────▶│  Supabase   │
│   (React)   │         │  Backend     │         │  PostgreSQL │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │  RAG System  │
                        │  (Llama 3.2) │
                        └──────────────┘
```

- **Frontend**: React.js with Vite, Tailwind CSS
- **Backend**: FastAPI (Python)
- **Database**: Supabase (PostgreSQL with vector extensions)
- **RAG**: Llama 3.2 3B Instruct (local inference)
- **Embeddings**: Sentence Transformers

## Features

### User Features
- OTP-based phone number authentication
- Chat with AI assistant (RAG-powered)
- Automatic ticket creation when AI confidence is low
- Request human agent assistance
- View complete chat history (WhatsApp-style)
- Urdu/English language support

### Agent Features
- View active, in-progress, and resolved tickets
- Real-time chat with users
- Assign and resolve tickets
- Domain-based ticket routing
- Optimistic UI updates for instant feedback

### Admin Features
- Comprehensive analytics dashboard
- Query statistics (daily, monthly, yearly)
- RAG vs Human agent comparison
- Ticket resolution time analytics
- Domain-wise statistics
- Real-time ticket monitoring

## 📦 Prerequisites

### Backend
- Python 3.8+
- pip
- Supabase account and project
- Knowledge base ZIP file (`Al Khidmat Knowledge Base.zip`)
- LLM model file (`Llama-3.2-3B-Instruct-Q4_K_M.gguf`)

### Frontend
- Node.js 18+
- npm or yarn

## 🔧 Backend Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FYP-Al-Khidmat-Public-Chat-Portal
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For CPU-only PyTorch (smaller, faster install):
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
# Or use SUPABASE_ANON_KEY instead

# Knowledge Base Path (optional, defaults to "Al Khidmat Knowledge Base.zip")
ALKHIDMAT_ZIP_PATH=Al Khidmat Knowledge Base.zip
```

### 5. Prepare Knowledge Base

Ensure the knowledge base ZIP file is in the root directory:
- `Al Khidmat Knowledge Base.zip`

### 6. Prepare LLM Model

Download and place the LLM model file in the root directory:
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf`

You can download it from [Hugging Face](https://huggingface.co/models) or other sources.

### 7. Set Up Supabase Database

1. Create a new Supabase project
2. Run the SQL schema (see `chat_session.sql` or database documentation)
3. Ensure vector extension is enabled:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### 8. Run the Backend Server

```bash
python -m api.main
```
#incase this creates problems then run this command :


Or using uvicorn directly:

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## 🎨 Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables (Optional)

Create a `.env` file in the `frontend` directory:

```env
VITE_API_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`.

### 4. Run the Frontend

**Development Mode:**
```bash
npm run dev
```

The frontend will start on `http://localhost:3000` (or the next available port).

**Production Build:**
```bash
npm run build
npm run preview
```

### Running the evaluation 
python test_rag_evaluation.py --max-cases 3 --use-openai-judge
#you can determine the number of tests you are going to run this for, also the file has path to english_testcases only for now, will add the other two soon.


## 🔐 Environment Variables

### Backend (.env in root)

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SUPABASE_URL` | Your Supabase project URL | Yes | - |
| `SUPABASE_KEY` or `SUPABASE_ANON_KEY` | Supabase anonymous key | Yes | - |
| `ALKHIDMAT_ZIP_PATH` | Path to knowledge base ZIP | No | `Al Khidmat Knowledge Base.zip` |

### Frontend (.env in frontend/)

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `VITE_API_URL` | Backend API URL | No | `http://localhost:8000` |

## 🚀 Running the Application

### Complete Setup

1. **Start Backend** (Terminal 1):
   ```bash
   cd FYP-Al-Khidmat-Public-Chat-Portal
   source venv/bin/activate  # If using virtual environment
   python api_server.py
   ```

2. **Start Frontend** (Terminal 2):
   ```bash
   cd FYP-Al-Khidmat-Public-Chat-Portal/frontend
   npm run dev
   ```

3. **Access the Application**:
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

### First-Time Setup Notes

- On first run, the RAG system will build the knowledge base index (this may take several minutes)
- The index is cached and won't rebuild on subsequent runs
- Ensure you have sufficient disk space for the index and model files

## 📁 Project Structure

```
FYP-Al-Khidmat-Public-Chat-Portal/
├── api_server.py              # FastAPI backend server
├── RAG_supabase.py            # RAG pipeline implementation
├── db_operations.py            # Database operations
├── supabase_client.py          # Supabase client initialization
├── domain_anchors.py           # Domain classification
├── requirements.txt            # Python dependencies
├── .env                        # Backend environment variables
├── README.md                   # This file
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Chatbot.jsx
│   │   │   ├── ChatMessages.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   └── Spinner.jsx
│   │   ├── pages/              # Page components
│   │   │   ├── Welcome.jsx
│   │   │   ├── UserLogin.jsx
│   │   │   ├── AgentLogin.jsx
│   │   │   ├── AdminLogin.jsx
│   │   │   ├── AgentDashboard.jsx
│   │   │   └── AdminDashboard.jsx
│   │   ├── hooks/             # Custom React hooks
│   │   ├── api.js             # API client
│   │   ├── App.jsx            # Main app component
│   │   └── main.jsx           # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── alkhidmat_index/            # Generated RAG index (created on first run)
├── Al Khidmat Knowledge Base.zip  # Knowledge base source
└── Llama-3.2-3B-Instruct-Q4_K_M.gguf  # LLM model file
```

## 🔌 API Endpoints

### Authentication

- `POST /auth/user/send-otp` - Send OTP to user phone number
- `POST /auth/user/verify-otp` - Verify OTP and create session
- `POST /auth/agent/login` - Agent login (email/password)
- `POST /auth/admin/login` - Admin login (email/password)

### Chat

- `POST /chats` - Create/get chat session
- `POST /chats/{session_id}/chat` - Send message and get RAG response
- `GET /chats/{session_id}/history` - Get chat history for session
- `POST /chats/{session_id}/request-human` - Request human agent assistance

### Tickets (Agent)

- `GET /agent/tickets` - List tickets (with optional status filter)
- `GET /agent/tickets/{ticket_id}/chat` - Get chat messages for ticket
- `POST /agent/tickets/{ticket_id}/assign` - Assign ticket to agent
- `POST /agent/tickets/{ticket_id}/resolve` - Resolve ticket
- `POST /agent/tickets/{ticket_id}/message` - Send message as agent

### Analytics (Admin)

- `GET /admin/analytics` - Get comprehensive analytics
- `GET /admin/tickets` - List all tickets (with optional filter)

See `http://localhost:8000/docs` for interactive API documentation.

## 🗄️ Database Schema

The application uses Supabase (PostgreSQL) with the following main tables:

- `users` - User accounts (phone number based)
- `sessions` - Chat sessions
- `queries` - User queries
- `responses` - AI/Agent responses
- `tickets` - Support tickets
- `human_agents` - Agent accounts
- `admins` - Admin accounts
- `documents` - Knowledge base documents (with embeddings)
- `response_documents` - Links responses to source documents

See database documentation or SQL schema files for detailed structure.

## 🖥️ GPU Acceleration (Mac)

For Apple Silicon Macs, GPU acceleration is automatically enabled. See `GPU_SETUP_MAC.md` for detailed instructions.

**Quick Setup:**
```bash
# Install llama-cpp-python with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

The system automatically detects Apple Silicon and enables GPU acceleration.

### Performance Issues

**Issue: Slow RAG responses**
- Solution: Enable GPU acceleration (see GPU setup), or use a smaller model

**Issue: High memory usage**
- Solution: Use CPU-only PyTorch, reduce batch sizes, or use a quantized model

## 📚 Additional Documentation

- [Frontend README](frontend/README.md) - Detailed frontend documentation
- [GPU Setup Guide](GPU_SETUP_MAC.md) - GPU acceleration setup for Mac
- [Literature Review](https://docs.google.com/document/d/10z9Gnw25w7gN_Anj4ITdmSic1zdSyWOAnEbx52THtEw/edit?usp=sharing)
- [RAG Literature Review](https://docs.google.com/document/d/1SzWvL0Rj1IjQTViIihLP06kh3rGxI09UANCHIo0LhI8/edit?usp=sharing)

## 👥 User Roles

### User
- Phone number authentication via OTP
- Chat with AI assistant
- Request human agent help
- View chat history

### Agent
- Email/password authentication
- View and manage tickets
- Chat with users
- Assign and resolve tickets

### Admin
- Email/password authentication
- View analytics and statistics
- Monitor all tickets
- System insights

## 🔒 Security Notes

- In production, replace in-memory session storage with Redis or JWT tokens
- Use environment variables for all sensitive credentials
- Enable HTTPS in production
- Implement rate limiting for API endpoints
- Use Supabase Row Level Security (RLS) policies

**Note**: This is a Final Year Project (FYP) for Alkhidmat Foundation. For questions or issues, please refer to the project documentation or contact the development team.
