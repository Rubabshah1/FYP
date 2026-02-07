# Alkhidmat Chat Portal - Complete Workflow Documentation

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Database Schema & Operations](#database-schema--operations)
3. [Backend API Endpoints](#backend-api-endpoints)
4. [RAG System Workflow](#rag-system-workflow)
5. [Frontend Components & Workflows](#frontend-components--workflows)
6. [User Flows](#user-flows)
7. [API Call Mappings](#api-call-mappings)
8. [Data Flow Diagrams](#data-flow-diagrams)

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────┐
│   Frontend      │  React.js + Vite + Tailwind CSS
│   (React)       │
└────────┬────────┘
         │ HTTP/REST API
         │
┌────────▼────────┐
│   Backend API   │  FastAPI (Python)
│   (FastAPI)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Supabase│ │  RAG  │
│  DB    │ │System │
└────────┘ └───────┘
```

### Technology Stack

**Frontend:**
- React.js 18+ with Hooks
- Vite.js (Build Tool)
- Tailwind CSS (Styling)
- Recharts (Charts/Graphs)
- React Router (Client-side routing)

**Backend:**
- FastAPI (Python Web Framework)
- Supabase Python Client (Database)
- Uvicorn (ASGI Server)

**RAG System:**
- Sentence Transformers (Embeddings)
- Llama-cpp-python (Local LLM)
- Supabase pgvector (Vector Search)
- LangChain Text Splitter (Chunking)

**Database:**
- Supabase (PostgreSQL + pgvector)
- Tables: users, sessions, queries, responses, tickets, human_agents, admins, documents

---

## Database Schema & Operations

### Database Tables

#### 1. `users`
- **Purpose**: Store user accounts (phone-based authentication)
- **Fields**:
  - `id` (UUID, Primary Key)
  - `phone_number` (Text, Unique)
  - `created_at` (Timestamp)

#### 2. `sessions`
- **Purpose**: Track chat sessions for users
- **Fields**:
  - `session_id` (UUID, Primary Key)
  - `user_id` (UUID, Foreign Key → users.id)
  - `started_at` (Timestamp)
  - `last_active` (Timestamp)

#### 3. `queries`
- **Purpose**: Store user questions/queries
- **Fields**:
  - `query_id` (UUID, Primary Key)
  - `session_id` (UUID, Foreign Key → sessions.session_id)
  - `domain` (Text) - Classified domain (donation, healthcare, general)
  - `timestamp` (Timestamp)
  - `content` (Text) - The actual query text

#### 4. `responses`
- **Purpose**: Store bot/agent responses
- **Fields**:
  - `response_id` (UUID, Primary Key)
  - `session_id` (UUID, Foreign Key → sessions.session_id)
  - `confidence` (Numeric) - RAG confidence score (0-1)
  - `domain` (Text) - Response domain
  - `timestamp` (Timestamp)
  - `content` (Text) - The response text

#### 5. `tickets`
- **Purpose**: Track support tickets (when RAG confidence is low or user requests agent)
- **Fields**:
  - `ticket_id` (UUID, Primary Key)
  - `response_id` (UUID, Foreign Key → responses.response_id, Unique)
  - `created_at` (Timestamp)
  - `resolved_at` (Timestamp, Nullable)
  - `status` (Text) - 'active', 'in_progress', 'resolved'
  - `agent_id` (UUID, Foreign Key → human_agents.agent_id, Nullable)

#### 6. `human_agents`
- **Purpose**: Store human agent accounts
- **Fields**:
  - `agent_id` (UUID, Primary Key)
  - `first_name` (Text)
  - `last_name` (Text)
  - `email` (Text, Unique)
  - `password` (Text) - Hashed password
  - `domain` (Text) - Agent specialization (donation, healthcare, general)

#### 7. `admins`
- **Purpose**: Store admin accounts
- **Fields**:
  - `admin_id` (UUID, Primary Key)
  - `first_name` (Text)
  - `last_name` (Text)
  - `email` (Text, Unique)
  - `password` (Text) - Hashed password

#### 8. `documents`
- **Purpose**: Store knowledge base chunks with embeddings
- **Fields**:
  - `doc_id` (UUID, Primary Key)
  - `chunk_text` (Text)
  - `chunk_index` (Integer)
  - `category` (Text)
  - `filename` (Text)
  - `file_path` (Text)
  - `doc_domain` (Text)
  - `embedding` (Vector[768]) - pgvector embedding
  - `created_at` (Timestamp)
  - `last_updated` (Timestamp)

#### 9. `response_documents`
- **Purpose**: Link responses to source documents (many-to-many)
- **Fields**:
  - `id` (UUID, Primary Key)
  - `response_id` (UUID, Foreign Key → responses.response_id)
  - `doc_id` (UUID, Foreign Key → documents.doc_id)
  - `relevance_score` (Float)
  - `rank_position` (Integer)
  - `created_at` (Timestamp)

### Key Database Operations (`db_operations.py`)

#### User Operations
- `create_user(phone_number)` - Create new user account
- `get_user_by_phone(phone_number)` - Retrieve user by phone
- `get_user(user_id)` - Get user by ID

#### Session Operations
- `create_session(user_id)` - Create new chat session
- `get_session(session_id)` - Get session details
- `update_session_activity(session_id)` - Update last_active timestamp
- `get_user_chat_history(user_id)` - Get all messages across all sessions for a user
- `get_session_chat_history(session_id)` - Get messages for a specific session

#### Query Operations
- `create_query(session_id, content)` - Store user query
- `update_query_domain(query_id, domain)` - Update query domain after classification

#### Response Operations
- `create_response(session_id, content, confidence, domain)` - Store bot/agent response

#### Ticket Operations
- `create_ticket(response_id, domain)` - Create ticket, auto-assign to domain-matched agent
- `get_ticket(ticket_id)` - Get ticket details
- `assign_ticket(ticket_id, agent_id)` - Assign ticket to agent
- `resolve_ticket(ticket_id)` - Mark ticket as resolved
- `list_tickets(status, agent_id)` - List tickets with filters

#### Analytics Operations
- `get_ticket_analytics()` - Comprehensive analytics for admin dashboard
  - Total queries, RAG vs Human answered
  - Time-based statistics (daily, monthly, yearly)
  - Ticket statistics
  - Domain statistics

---

## Backend API Endpoints

### Base URL
- Development: `http://localhost:8000`
- Production: Set via environment variable

### Authentication Endpoints

#### User Authentication (OTP-based)

**POST `/auth/user/send-otp`**
- **Purpose**: Send OTP to user's phone number
- **Request Body**:
  ```json
  {
    "phone_number": "+923001234567"
  }
  ```
- **Response**:
  ```json
  {
    "message": "OTP sent successfully",
    "otp": "123456"  // Only in development
  }
  ```
- **Database**: Creates/updates OTP in memory store (in production, use Redis)

**POST `/auth/user/verify-otp`**
- **Purpose**: Verify OTP and create/login user
- **Request Body**:
  ```json
  {
    "phone_number": "+923001234567",
    "otp": "123456"
  }
  ```
- **Response**:
  ```json
  {
    "session_id": "uuid-here",
    "user_id": "uuid-here",
    "chat_history": [...]  // All previous messages
  }
  ```
- **Database**: 
  - Creates user if doesn't exist
  - Creates new session
  - Returns full chat history

#### Agent Authentication

**POST `/auth/agent/login`**
- **Purpose**: Agent login with email/password
- **Request Body**:
  ```json
  {
    "email": "agent@example.com",
    "password": "password123"
  }
  ```
- **Response**:
  ```json
  {
    "agent_token": "uuid-session-id",
    "agent_id": "uuid-here",
    "agent": {...}
  }
  ```
- **Database**: Validates credentials, creates session

#### Admin Authentication

**POST `/auth/admin/login`**
- **Purpose**: Admin login with email/password
- **Request Body**:
  ```json
  {
    "email": "admin@example.com",
    "password": "password123"
  }
  ```
- **Response**:
  ```json
  {
    "admin_token": "uuid-session-id",
    "admin_id": "uuid-here",
    "admin": {...}
  }
  ```
- **Database**: Validates credentials, creates session

### Chat Endpoints

**POST `/chats`**
- **Purpose**: Create/get chat session and return history
- **Headers**: `X-Session-ID: <session_id>`
- **Response**:
  ```json
  {
    "session_id": "uuid-here",
    "db_session_id": "uuid-here",
    "user_id": "uuid-here",
    "chat_history": [...]  // All user's messages
  }
  ```
- **Database**: Returns user's full chat history across all sessions

**GET `/chats/{session_id}/history`**
- **Purpose**: Get chat history for specific session
- **Headers**: `X-Session-ID: <session_id>`
- **Response**:
  ```json
  {
    "session_id": "uuid-here",
    "messages": [
      {
        "role": "user",
        "content": "...",
        "timestamp": "..."
      },
      {
        "role": "assistant",
        "content": "...",
        "timestamp": "..."
      }
    ]
  }
  ```

**POST `/chats/{session_id}`**
- **Purpose**: Send message and get RAG response
- **Headers**: `X-Session-ID: <session_id>`
- **Request Body**:
  ```json
  {
    "message": "What donation methods does Alkhidmat accept?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Alkhidmat accepts donations via...",
    "sources": [...],
    "agent_chat": false,
    "response_id": "uuid-here",
    "confidence": 0.85,
    "ticket_id": null
  }
  ```
- **Workflow**:
  1. Creates query record
  2. Checks if user requested human agent
  3. If yes → Creates ticket, returns agent routing message
  4. If no → Processes with RAG
  5. Checks confidence threshold (0.80)
  6. If low → Creates ticket, returns routing message
  7. If high → Returns RAG answer
  8. Stores response in database

### Ticket Endpoints

**POST `/tickets`**
- **Purpose**: Create ticket manually (from frontend button)
- **Headers**: `X-Session-ID: <session_id>`
- **Request Body**:
  ```json
  {
    "session_id": "uuid-here",
    "initial_message": "User requested human agent"
  }
  ```
- **Response**:
  ```json
  {
    "ticket_id": "uuid-here",
    "status": "active",
    "response_id": "uuid-here"
  }
  ```

**GET `/tickets`**
- **Purpose**: List tickets (for agents)
- **Headers**: `X-Agent-Token: <token>`
- **Query Params**: `?status=active&unassigned=true`
- **Response**:
  ```json
  {
    "tickets": [...]
  }
  ```

**POST `/tickets/{ticket_id}/assign`**
- **Purpose**: Assign ticket to agent
- **Headers**: `X-Agent-Token: <token>`
- **Response**:
  ```json
  {
    "ticket_id": "uuid-here",
    "status": "in_progress",
    "agent_id": "uuid-here"
  }
  ```

**POST `/tickets/{ticket_id}/resolve`**
- **Purpose**: Resolve ticket
- **Headers**: `X-Agent-Token: <token>`
- **Response**:
  ```json
  {
    "ticket_id": "uuid-here",
    "status": "resolved",
    "resolved_at": "..."
  }
  ```

**GET `/tickets/{ticket_id}/chat`**
- **Purpose**: Get chat messages for a ticket
- **Headers**: `X-Agent-Token: <token>`
- **Response**:
  ```json
  {
    "ticket_id": "uuid-here",
    "messages": [
      {
        "type": "query",
        "role": "user",
        "sender": "user",
        "content": "...",
        "timestamp": "..."
      },
      {
        "type": "response",
        "role": "assistant",
        "sender": "assistant",
        "content": "...",
        "timestamp": "..."
      },
      {
        "type": "response",
        "role": "agent",
        "sender": "agent",
        "content": "...",
        "timestamp": "..."
      }
    ]
  }
  ```

**POST `/tickets/{ticket_id}/message`**
- **Purpose**: Agent sends message to user
- **Headers**: `X-Agent-Token: <token>`
- **Request Body**:
  ```json
  {
    "message": "Hello, how can I help you?",
    "sender": "agent"
  }
  ```
- **Response**:
  ```json
  {
    "status": "sent",
    "ticket_id": "uuid-here",
    "response_id": "uuid-here",
    "message": "Hello, how can I help you?"
  }
  ```
- **Database**: Stores agent message as response with confidence=1.0

### Admin Endpoints

**GET `/admin/analytics`**
- **Purpose**: Get comprehensive analytics
- **Headers**: `X-Admin-Token: <token>`
- **Response**:
  ```json
  {
    "total_queries": 150,
    "total_rag_answered": 120,
    "total_human_answered": 30,
    "total_tickets": 30,
    "active_tickets": 5,
    "in_progress_tickets": 10,
    "resolved_tickets": 15,
    "average_resolution_time_seconds": 3600,
    "queries_over_time": {
      "daily": [...],
      "monthly": [...],
      "yearly": [...]
    },
    "rag_vs_human": {
      "rag_responses": 120,
      "human_responses": 30,
      "rag_percentage": 80.0,
      "human_percentage": 20.0,
      "avg_rag_confidence": 0.75
    }
  }
  ```

**GET `/admin/tickets`**
- **Purpose**: List all tickets (admin view)
- **Headers**: `X-Admin-Token: <token>`
- **Query Params**: `?status=active`
- **Response**:
  ```json
  {
    "tickets": [...]
  }
  ```

---

## RAG System Workflow

### RAG Components (`RAG_supabase.py`)

#### 1. Domain Classification
- **Class**: `DomainClassifier`
- **Purpose**: Classify queries into domains (donation, healthcare, general)
- **Method**: 
  - Uses pre-computed domain centroids from anchor queries
  - Calculates cosine similarity between query embedding and domain centroids
  - Returns domain with highest similarity

#### 2. Confidence Scoring
- **Class**: `ConfidenceScorer`
- **Purpose**: Calculate confidence scores for RAG responses
- **Metrics**:
  - Retrieval Confidence (cosine similarity with documents)
  - Average Token Confidence (from LLM log probabilities)
  - Weighted Top-K Confidence
  - Perplexity
  - Entropy Confidence
- **Combined Score**: Weighted average (50% retrieval, 25% token, 25% top-k)

#### 3. Document Retrieval
- **Function**: `retrieve_from_supabase(query, top_k=5)`
- **Process**:
  1. Encode query with embedding model (`intfloat/multilingual-e5-base`)
  2. Call Supabase RPC function `match_documents` for vector search
  3. Fetch document embeddings for confidence scoring
  4. Return top-k relevant chunks with similarity scores

#### 4. Answer Generation
- **Function**: `generate_answer(query, top_k=5)`
- **Process**:
  1. **Domain Classification**: Classify query domain
  2. **Retrieval**: Get relevant documents from Supabase
  3. **Context Building**: Combine retrieved chunks
  4. **Prompt Construction**: Build prompt with context and query
  5. **LLM Generation**: Generate answer using Llama-cpp-python
  6. **Confidence Calculation**: Calculate all confidence metrics
  7. **Translation**: Translate to Urdu if query was in Urdu
  8. **Return**: Answer, sources, confidence scores, domain classification

### RAG Workflow in API

**When `/chats/{session_id}` is called:**

```
1. User sends message
   ↓
2. Create query record in database
   ↓
3. Check if user requested human agent
   ├─ YES → Create ticket, return routing message
   └─ NO → Continue to RAG
   ↓
4. RAG Processing:
   ├─ Domain Classification
   ├─ Document Retrieval (vector search)
   ├─ LLM Answer Generation
   └─ Confidence Calculation
   ↓
5. Check Confidence Threshold (0.80)
   ├─ LOW (< 0.80) → Create ticket, return routing message
   └─ HIGH (>= 0.80) → Return RAG answer
   ↓
6. Store response in database
   ↓
7. Update query domain in database
   ↓
8. Return response to frontend
```

---

## Frontend Components & Workflows

### Component Structure

```
App.jsx
├── Welcome.jsx (Landing page - role selection)
├── UserLogin.jsx (OTP login)
├── Chatbot.jsx (Main chat interface)
│   ├── ChatMessages.jsx (Message display)
│   └── ChatInput.jsx (Input field)
├── AgentLogin.jsx
├── AgentDashboard.jsx
│   └── Ticket chat interface
├── AdminLogin.jsx
└── AdminDashboard.jsx
    └── Analytics charts
```

### Key Frontend Files

#### `frontend/src/App.jsx`
- **Purpose**: Main routing and authentication logic
- **Routes**:
  - `/` → Welcome page
  - `/user` → User login
  - `/user/chat` → Chat interface
  - `/agent/login` → Agent login
  - `/agent/dashboard` → Agent dashboard
  - `/admin/login` → Admin login
  - `/admin/dashboard` → Admin dashboard

#### `frontend/src/components/Chatbot.jsx`
- **Purpose**: Main chat interface for users
- **State**:
  - `sessionId` - Current session ID
  - `messages` - Chat messages array
  - `isAgentChat` - Whether chatting with human agent
  - `ticketId` - Current ticket ID (if in agent chat)
- **Key Functions**:
  - `loadChatHistory()` - Load previous messages
  - `submitNewMessage()` - Send message and get response
  - `requestHumanAgent()` - Create ticket for human agent
- **Polling**: Polls every 3 seconds when `isAgentChat` is true

#### `frontend/src/components/ChatMessages.jsx`
- **Purpose**: Display chat messages (WhatsApp-style)
- **Features**:
  - User messages (green bubbles, right-aligned)
  - Assistant messages (white bubbles, left-aligned)
  - Agent messages (white bubbles, left-aligned)
  - Loading indicator (typing dots)
  - Auto-scroll to latest message

#### `frontend/src/pages/AgentDashboard.jsx`
- **Purpose**: Agent interface for managing tickets
- **Features**:
  - Ticket list (active, in_progress, resolved)
  - Chat interface for each ticket
  - Assign/Resolve ticket buttons
  - Message sending
  - Chat caching (prevents refetching)
  - Optimistic updates

#### `frontend/src/pages/AdminDashboard.jsx`
- **Purpose**: Admin analytics dashboard
- **Features**:
  - Statistics cards (total queries, RAG vs Human)
  - Time-based charts (daily, monthly, yearly)
  - Pie chart (RAG vs Human)
  - Ticket list
  - Performance metrics

### Frontend API Client (`frontend/src/api.js`)

All API calls are centralized in `api.js`:

- `sendOTP(phoneNumber)` → `POST /auth/user/send-otp`
- `verifyOTP(phoneNumber, otp)` → `POST /auth/user/verify-otp`
- `createChat()` → `POST /chats`
- `sendChatMessage(sessionId, message)` → `POST /chats/{session_id}`
- `getChatHistory(sessionId)` → `GET /chats/{session_id}/history`
- `createTicket(sessionId, message)` → `POST /tickets`
- `listTickets(status)` → `GET /tickets`
- `assignTicket(ticketId)` → `POST /tickets/{ticket_id}/assign`
- `resolveTicket(ticketId)` → `POST /tickets/{ticket_id}/resolve`
- `getTicketChat(ticketId)` → `GET /tickets/{ticket_id}/chat`
- `sendAgentMessage(ticketId, message)` → `POST /tickets/{ticket_id}/message`
- `getAnalytics()` → `GET /admin/analytics`
- `adminListTickets(status)` → `GET /admin/tickets`

---

## User Flows

### Flow 1: User Chat Flow (RAG)

```
1. User lands on Welcome page
   ↓
2. Selects "User" → Redirects to UserLogin
   ↓
3. Enters phone number → API: POST /auth/user/send-otp
   ↓
4. Enters OTP → API: POST /auth/user/verify-otp
   ├─ Creates user (if new)
   ├─ Creates session
   └─ Returns chat_history
   ↓
5. Redirects to Chatbot component
   ↓
6. Loads chat history → API: POST /chats (or from localStorage)
   ↓
7. User types message and sends
   ↓
8. Frontend: Adds user message optimistically
   Frontend: Adds loading assistant message
   ↓
9. API: POST /chats/{session_id}
   ├─ Creates query record
   ├─ Processes with RAG
   ├─ Checks confidence threshold
   ├─ Stores response
   └─ Returns answer
   ↓
10. Frontend: Updates assistant message with answer
    ↓
11. If confidence low → Ticket created automatically
    Frontend: Shows "routing to agent" message
    Sets isAgentChat = true
    Starts polling for agent messages
```

### Flow 2: User Requests Human Agent

```
1. User clicks "Need Human Help?" button
   OR types "connect me with agent"
   ↓
2. Frontend: API: POST /tickets
   OR
   API: POST /chats/{session_id} (detects agent request)
   ↓
3. Backend: Creates ticket
   ├─ Creates placeholder response
   ├─ Creates ticket
   ├─ Auto-assigns to domain-matched agent
   └─ Returns ticket_id
   ↓
4. Frontend: Sets isAgentChat = true
   Shows "connecting to agent" message
   Starts polling every 3 seconds
   ↓
5. Agent receives ticket in dashboard
   ↓
6. Agent assigns ticket → API: POST /tickets/{ticket_id}/assign
   ↓
7. Agent sends message → API: POST /tickets/{ticket_id}/message
   ├─ Stores message as response (confidence=1.0)
   └─ Returns success
   ↓
8. User's polling detects new message
   API: POST /chats (returns updated chat_history)
   ↓
9. Frontend: Displays agent message
```

### Flow 3: Agent Dashboard Flow

```
1. Agent logs in → API: POST /auth/agent/login
   ↓
2. Redirects to AgentDashboard
   ↓
3. Loads tickets → API: GET /tickets
   ├─ Filters locally (active, in_progress, resolved)
   └─ Displays in tabs
   ↓
4. Agent selects ticket
   ↓
5. Loads chat → API: GET /tickets/{ticket_id}/chat
   ├─ Checks cache first
   ├─ If cached → Shows immediately
   └─ Fetches from API in background
   ↓
6. Agent types message and sends
   ↓
7. Frontend: Adds message optimistically
   ↓
8. API: POST /tickets/{ticket_id}/message
   ├─ Stores message as response
   └─ Returns success
   ↓
9. Frontend: Refreshes chat after 500ms
   ↓
10. Agent resolves ticket → API: POST /tickets/{ticket_id}/resolve
    Frontend: Updates ticket status optimistically
```

### Flow 4: Admin Dashboard Flow

```
1. Admin logs in → API: POST /auth/admin/login
   ↓
2. Redirects to AdminDashboard
   ↓
3. Loads analytics → API: GET /admin/analytics
   ├─ Total queries, RAG vs Human stats
   ├─ Time-based data (daily, monthly, yearly)
   └─ Ticket statistics
   ↓
4. Loads tickets → API: GET /admin/tickets
   ↓
5. Displays:
   ├─ Statistics cards
   ├─ Charts (Line chart, Pie chart)
   └─ Ticket list
   ↓
6. Admin can filter tickets by status
   ↓
7. Admin can click ticket to view details
```

---

## API Call Mappings

### User Actions → API Calls

| User Action | Frontend Component | API Call | Backend Endpoint | Database Operations |
|------------|-------------------|----------|------------------|-------------------|
| Enter phone number | UserLogin.jsx | `sendOTP()` | `POST /auth/user/send-otp` | Stores OTP in memory |
| Verify OTP | UserLogin.jsx | `verifyOTP()` | `POST /auth/user/verify-otp` | Creates user, creates session, returns chat_history |
| Load chat history | Chatbot.jsx | `createChat()` | `POST /chats` | Returns user's full chat history |
| Send message | Chatbot.jsx | `sendChatMessage()` | `POST /chats/{session_id}` | Creates query, processes RAG, creates response, creates ticket (if low confidence) |
| Request human agent (button) | Chatbot.jsx | `createTicket()` | `POST /tickets` | Creates response, creates ticket |
| Request human agent (text) | Chatbot.jsx | `sendChatMessage()` | `POST /chats/{session_id}` | Detects agent request, creates ticket |
| Poll for new messages | Chatbot.jsx | `createChat()` | `POST /chats` | Returns updated chat_history (every 3s when isAgentChat=true) |

### Agent Actions → API Calls

| Agent Action | Frontend Component | API Call | Backend Endpoint | Database Operations |
|------------|-------------------|----------|------------------|-------------------|
| Login | AgentLogin.jsx | `agentLogin()` | `POST /auth/agent/login` | Validates credentials, creates session |
| Load tickets | AgentDashboard.jsx | `listTickets()` | `GET /tickets` | Returns all tickets for agent |
| Select ticket | AgentDashboard.jsx | `getTicketChat()` | `GET /tickets/{ticket_id}/chat` | Returns session messages (queries + responses) |
| Assign ticket | AgentDashboard.jsx | `assignTicket()` | `POST /tickets/{ticket_id}/assign` | Updates ticket status to 'in_progress', sets agent_id |
| Send message | AgentDashboard.jsx | `sendAgentMessage()` | `POST /tickets/{ticket_id}/message` | Creates response (confidence=1.0) |
| Resolve ticket | AgentDashboard.jsx | `resolveTicket()` | `POST /tickets/{ticket_id}/resolve` | Updates ticket status to 'resolved', sets resolved_at |

### Admin Actions → API Calls

| Admin Action | Frontend Component | API Call | Backend Endpoint | Database Operations |
|------------|-------------------|----------|------------------|-------------------|
| Login | AdminLogin.jsx | `adminLogin()` | `POST /auth/admin/login` | Validates credentials, creates session |
| Load analytics | AdminDashboard.jsx | `getAnalytics()` | `GET /admin/analytics` | Aggregates queries, responses, tickets, calculates statistics |
| Load tickets | AdminDashboard.jsx | `adminListTickets()` | `GET /admin/tickets` | Returns all tickets |
| Filter tickets | AdminDashboard.jsx | `adminListTickets(status)` | `GET /admin/tickets?status=active` | Returns filtered tickets |

---

## Data Flow Diagrams

### Complete Message Flow (User → RAG → Response)

```
┌─────────┐
│  User   │ Types: "What donation methods?"
└────┬────┘
     │
     ▼
┌─────────────────┐
│  Frontend       │ 1. Adds user message optimistically
│  Chatbot.jsx    │ 2. Adds loading assistant message
└────┬────────────┘
     │
     │ POST /chats/{session_id}
     │ { "message": "What donation methods?" }
     ▼
┌─────────────────┐
│  Backend API    │ 1. Creates query record
│  api_server.py  │ 2. Checks for agent request
└────┬────────────┘ 3. Processes with RAG
     │
     │ generate_answer()
     ▼
┌─────────────────┐
│  RAG System     │ 1. Domain Classification
│  RAG_supabase.py│ 2. Document Retrieval (vector search)
└────┬────────────┘ 3. LLM Generation
     │             4. Confidence Calculation
     │
     │ retrieve_from_supabase()
     ▼
┌─────────────────┐
│  Supabase DB    │ Vector search via pgvector
│  documents      │ Returns top-k chunks
└────┬────────────┘
     │
     │ Returns chunks
     ▼
┌─────────────────┐
│  RAG System     │ Builds prompt, generates answer
│  Llama-cpp      │ Calculates confidence
└────┬────────────┘
     │
     │ Returns answer + confidence
     ▼
┌─────────────────┐
│  Backend API    │ 1. Checks confidence threshold (0.80)
│  api_server.py  │ 2. If low → Creates ticket
└────┬────────────┘ 3. Stores response
     │             4. Updates query domain
     │
     │ Returns response
     ▼
┌─────────────────┐
│  Frontend       │ Updates assistant message
│  Chatbot.jsx    │ Displays answer
└─────────────────┘
```

### Ticket Creation & Agent Assignment Flow

```
┌─────────┐
│  User   │ Low confidence OR requests agent
└────┬────┘
     │
     ▼
┌─────────────────┐
│  Backend API    │ Creates ticket
│  api_server.py  │
└────┬────────────┘
     │
     │ create_ticket()
     ▼
┌─────────────────┐
│  db_operations  │ 1. Creates placeholder response
│                 │ 2. Creates ticket (status='active')
└────┬────────────┘ 3. Finds agents by domain
     │             4. Auto-assigns to available agent
     │
     │ Returns ticket
     ▼
┌─────────────────┐
│  Frontend       │ Sets isAgentChat=true
│  Chatbot.jsx    │ Starts polling
└─────────────────┘
     │
     │ Polling every 3s
     │ POST /chats
     ▼
┌─────────────────┐
│  Backend API    │ Returns updated chat_history
│                 │ Includes agent messages
└─────────────────┘
```

### Agent Message Flow

```
┌─────────┐
│  Agent  │ Types message in ticket chat
└────┬────┘
     │
     ▼
┌─────────────────┐
│  Frontend       │ Adds message optimistically
│  AgentDashboard │
└────┬────────────┘
     │
     │ POST /tickets/{ticket_id}/message
     │ { "message": "...", "sender": "agent" }
     ▼
┌─────────────────┐
│  Backend API    │ 1. Gets ticket → session_id
│  api_server.py  │ 2. Creates response (confidence=1.0)
└────┬────────────┘ 3. Stores in database
     │
     │ create_response()
     ▼
┌─────────────────┐
│  Supabase DB    │ Stores response
│  responses      │
└────┬────────────┘
     │
     │ Returns success
     ▼
┌─────────────────┐
│  Frontend       │ Refreshes chat after 500ms
│  AgentDashboard │ Updates cache
└─────────────────┘
     │
     │ (User polling detects new message)
     ▼
┌─────────────────┐
│  User Frontend  │ Displays agent message
│  Chatbot.jsx    │
└─────────────────┘
```

---

## Key Configuration

### Environment Variables

**Backend (`.env`):**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
GPT4ALL_MODEL=Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Frontend (`.env`):**
```
VITE_API_URL=http://localhost:8000
```

### Confidence Threshold
- **Default**: `0.80` (80%)
- **Location**: `api_server.py` → `CONFIDENCE_THRESHOLD = 0.80`
- **Behavior**: If RAG confidence < threshold → Creates ticket, routes to agent

### Polling Intervals
- **User chat (agent messages)**: 3 seconds
- **Agent dashboard (tickets)**: On mount and visibility change
- **Admin dashboard**: On mount and filter change

### Domain Classification
- **Domains**: `donation`, `healthcare`, `general`
- **Method**: Cosine similarity with domain centroids
- **Location**: `RAG_supabase.py` → `DomainClassifier`

---

## Error Handling

### Common Errors & Solutions

1. **401 Unauthorized**
   - **Cause**: Invalid/expired session token
   - **Frontend**: Redirects to login page
   - **Backend**: Returns 401 status

2. **404 Not Found**
   - **Cause**: Session/ticket not found
   - **Backend**: Returns 404 with error message

3. **500 Internal Server Error**
   - **Cause**: RAG processing error, database error
   - **Backend**: Logs error, returns 500
   - **Frontend**: Shows error message

4. **Translation Timeout**
   - **Cause**: Google Translator API timeout (>15s)
   - **RAG**: Returns original English text
   - **Frontend**: Displays English answer

---

## Performance Optimizations

1. **Chat Caching**: Agent dashboard caches chat messages to prevent refetching
2. **Optimistic Updates**: UI updates immediately, syncs with server
3. **Request Deduplication**: Prevents concurrent API calls
4. **Local Filtering**: Ticket filtering done client-side to reduce API calls
5. **Polling Optimization**: Only polls when necessary (agent chat active)
6. **Embedding Reuse**: RAG reuses embedding model for domain classification

---

## Security Considerations

1. **Session Management**: UUID-based session tokens stored in database
2. **Password Hashing**: Agent/admin passwords hashed (currently plaintext for testing)
3. **OTP Storage**: In-memory (use Redis in production)
4. **CORS**: Configured for frontend origin only
5. **Input Validation**: Pydantic models validate all inputs

---

## Future Enhancements

1. **WebSocket Support**: Real-time messaging instead of polling
2. **Redis Integration**: For OTP storage and session management
3. **JWT Tokens**: Replace UUID session tokens
4. **Rate Limiting**: Prevent API abuse
5. **File Uploads**: Support document/image uploads
6. **Multi-language**: Support more languages beyond Urdu/English
7. **Voice Input**: Speech-to-text for queries
8. **Analytics Dashboard**: More detailed analytics and reports

---

## Troubleshooting

### RAG Not Responding
- Check if model file exists: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- Check Supabase connection
- Check if documents are indexed in database

### Messages Not Appearing
- Check browser console for errors
- Verify session token is valid
- Check database for stored messages

### Agent Messages Not Received
- Verify polling is active (`isAgentChat=true`)
- Check ticket status (should be 'in_progress')
- Verify agent message was stored (check database)

### Analytics Showing Zero
- Verify there's data in database (queries, responses, tickets)
- Check admin authentication token
- Check browser console for API errors

---

**Last Updated**: 2024
**Version**: 1.0

