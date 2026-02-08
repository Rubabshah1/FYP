# Alkhidmat Chat Portal - Complete Workflow Documentation

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Database Schema & Operations](#database-schema--operations)
3. [Backend API Endpoints](#backend-api-endpoints)
4. [RAG System Workflow](#rag-system-workflow)
5. [Enhanced Features](#enhanced-features)
6. [Frontend Components & Workflows](#frontend-components--workflows)
7. [User Flows](#user-flows)
8. [API Call Mappings](#api-call-mappings)
9. [Data Flow Diagrams](#data-flow-diagrams)

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
- Sentence Transformers (Embeddings) - `intfloat/multilingual-e5-base`
- Llama-cpp-python (Local LLM) - `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- Supabase pgvector (Vector Search)
- LangChain Text Splitter (Chunking)
- **Self-RAG Implementation** - Self-reflective RAG with answer verification
- **Multilingual Support** - English, Urdu script, and Roman Urdu
- **Brand Term Protection** - Preserves proper nouns during translation
- **Document Format Support** - Text (.txt), PDF (.pdf), and Word documents (.docx)

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

#### Knowledge Base Operations (`RAG_supabase.py`)
- `build_alkhidmat_rag(zip_path, clear_existing=False, incremental=False)` - Build/update knowledge base
  - **Incremental Mode**: When `incremental=True`, only processes new documents (skips existing ones)
  - **Full Rebuild**: When `incremental=False` and `clear_existing=True`, rebuilds entire KB
- `add_document_incremental(file_path, content, category, filename, reindex=False)` - Add single document incrementally
- `add_single_file_incremental(file_path, content, category, reindex=False)` - Helper for adding single file
- `add_documents_from_zip_incremental(zip_path, reindex_existing=False)` - Add documents from ZIP incrementally
- `document_exists(file_path)` - Check if document already exists in KB
- `delete_document_by_path(file_path)` - Delete document chunks for re-indexing
- `clear_documents_table()` - Clear all documents (use with caution!)

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
- **Session Token**: Uses database `session_id` (UUID) as the session token, ensuring persistence across server restarts

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

### Session Resolution

The system uses a robust session resolution mechanism to handle different session ID formats:

**Resolution Process** (`resolve_session_id()` helper):
1. **In-Memory Cache Check**: First checks if session_id exists in `user_sessions` dictionary
   - If found → Returns cached `db_session_id` immediately (fast path)
2. **UUID Validation**: Checks if session_id is a valid UUID format
   - If valid UUID → Looks up in database
   - If found in database → Restores to in-memory cache and returns `db_session_id`
   - If not found → Returns error: "Session not found in database. Please login again."
3. **Old Token Format**: If not a valid UUID
   - Returns error: "Session expired (old token format). Please login again to get a new session."

**Benefits**:
- Sessions persist across server restarts (stored in database)
- Fast lookup via in-memory cache
- Automatic cache restoration from database
- Handles both UUID and legacy token formats

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
- **Session Resolution**: Uses `resolve_session_id()` to handle UUID format, in-memory cache, and database lookup

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
- **Purpose**: Send message and get RAG response (with Self-RAG verification if enabled)
- **Headers**: `X-Session-ID: <session_id>` (preferred) or use URL parameter
- **Session Resolution**: Prefers header over URL parameter; resolves using `resolve_session_id()` helper
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
    "sources": [
      {
        "category": "Donation",
        "filename": "donation-methods.txt",
        "file_path": "Donation/donation-methods.txt",
        "similarity": 0.87
      }
    ],
    "agent_chat": false,
    "response_id": "uuid-here",
    "confidence": 0.85,
    "ticket_id": null
  }
  ```
- **Workflow**:
  1. Resolves session ID (handles UUID format, in-memory cache, database lookup)
  2. Updates session activity timestamp
  3. Creates query record in database
  4. Checks if user already has an active ticket for this session
  5. Checks if user requested human agent (comprehensive keyword detection):
     - Core keywords: "connect me with human/agent", "human agent", "talk to human/agent"
     - Pattern matching: checks for "connect", "i want", "i need", "can i", "let me" + agent keywords
     - Urdu keywords: "human se baat", "agent se baat", "human ko connect"
  6. If user requested agent:
     - If already has active ticket → Returns "already connected" message
     - If no active ticket → Creates ticket immediately with domain="general", returns agent routing message
     - Skips RAG processing entirely
  7. If no agent request → Processes with RAG:
     - **Self-RAG Path** (if `SELFRAG_ENABLE=True`):
       - Step 0: Domain relevance check
       - Step 1: Retrieval necessity check
       - Domain classification
       - Query expansion
       - Step 2: Document retrieval
       - Step 3: Document relevance assessment
       - Step 3.5: Answer in context check
       - Step 4: LLM answer generation
       - Step 5: Support verification
       - Step 6: Utility evaluation
       - Final confidence check
     - **Standard RAG Path** (if `SELFRAG_ENABLE=False`):
       - Domain classification
       - Query expansion
       - Document retrieval
       - LLM answer generation
       - Confidence calculation
  5. Checks if user already has an active ticket (prevents duplicate tickets)
  6. Checks confidence threshold (0.70 for auto-ticket creation)
  7. If low confidence AND no active ticket → Creates ticket automatically, returns routing message
     - Broadcasts ticket creation to agents via WebSocket
     - Translates routing message to Urdu if user query was in Urdu
  8. Stores response in database with confidence and domain
  9. Updates query domain in database
  10. Converts numpy types to native Python types for JSON serialization
  11. Returns response to frontend

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
    "status": "assigned",
    "ticket_id": "uuid-here",
    "agent_id": "uuid-here"
  }
  ```
- **WebSocket**: Broadcasts ticket assignment to all connected agents

**POST `/tickets/{ticket_id}/resolve`**
- **Purpose**: Resolve ticket
- **Headers**: `X-Agent-Token: <token>`
- **Response**:
  ```json
  {
    "status": "resolved",
    "ticket_id": "uuid-here"
  }
  ```
- **WebSocket**: Broadcasts ticket resolution to all connected agents

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
- **WebSocket**: Broadcasts message to user via WebSocket (`/ws/user/{session_id}`)

### WebSocket Endpoints

**WebSocket `/ws/user/{session_id}`**
- **Purpose**: Real-time messaging for users (receives agent messages)
- **Connection**: User connects with their session_id
- **Messages Received**:
  ```json
  {
    "type": "new_message",
    "message": {
      "role": "agent",
      "sender": "agent",
      "content": "...",
      "timestamp": "...",
      "response_id": "uuid-here"
    }
  }
  ```
- **Ping/Pong**: Supports ping messages for keep-alive
- **Auto-reconnect**: Frontend handles reconnection automatically

**WebSocket `/ws/agent/{agent_token}`**
- **Purpose**: Real-time ticket updates for agents
- **Connection**: Agent connects with their agent_token
- **Messages Received**:
  ```json
  {
    "type": "ticket_update",
    "ticket_id": "uuid-here",
    "update_type": "created" | "assigned" | "resolved",
    "data": {
      "ticket_id": "uuid-here",
      "status": "active" | "in_progress" | "resolved",
      "agent_id": "uuid-here" | null,
      "domain": "donation" | "healthcare" | "general"
    }
  }
  ```
- **Ping/Pong**: Supports ping messages for keep-alive
- **Auto-reconnect**: Frontend handles reconnection automatically

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

### Agentic AI Architecture (NEW)

The system now includes an **Agentic AI architecture** that regulates vector DB embeddings and reduces hallucinations through intelligent workflow control.

#### Agentic Components

**1. Router Agent** (`RouterAgent`)
- **Purpose**: Routes queries BEFORE embedding generation
- **Decisions**:
  - Is this chit-chat/greeting? → Return cached answer, skip embedding
  - Is retrieval needed? → Use Self-RAG retrieval necessity check
  - Can we answer without knowledge base? → Short-circuit to direct answer
- **Benefits**: Eliminates 30-60% of unnecessary embedding operations

**2. Retriever Agent** (`RetrieverAgent`)
- **Purpose**: Retrieves documents with automatic retry and query reformulation
- **Features**:
  - Checks average relevance after initial retrieval
  - If relevance < threshold → Reformulates query automatically
  - Retries retrieval with reformulated query (max 2 attempts)
  - Uses embedding cache for reformulated queries
- **Benefits**: Reduces missing-context hallucinations by improving retrieval quality

**3. Evidence Coverage Agent** (`EvidenceCoverageAgent`)
- **Purpose**: Verifies claim-by-claim that answers are supported by evidence
- **Process**:
  - Splits compound sentences into atomic claims (handles ", and", "and", commas, relative clauses)
  - For each claim: Checks if supported by context (allows multi-document support)
  - Removes unsupported claims automatically
  - Rejects answer if coverage < threshold
- **Features**:
  - **Compound Claim Splitting**: Breaks down complex sentences like "X provides A, B, and C" into separate claims
  - **Multi-Document Support**: Allows claims to be supported by information across multiple documents
  - **Summary Mode**: Relaxes strictness for general/about/introduction queries (70% threshold vs 80%)
  - **Smart Subject Inference**: Automatically adds subject to split claims for grammatical correctness
- **Benefits**: 
  - Prevents hallucinations like "Alkhidmat also offers X, Y, Z" when only X is documented
  - Handles multi-document summaries correctly (e.g., "About us" queries spanning multiple sources)
  - More accurate for overview/summary queries that naturally synthesize information

**4. Conversation Memory Agent** (`ConversationMemoryAgent`)
- **Purpose**: Manages conversation state for follow-up queries
- **Features**:
  - Detects follow-up queries (short queries, "what about", "also", etc.)
  - Stores: last domain, last document IDs, last confidence
  - Reuses previous context for follow-ups (no re-embedding needed)
- **Benefits**: Eliminates embedding for follow-ups, reduces hallucination on context switches

**5. Embedding Cache System**
- **Purpose**: Semantic deduplication of query embeddings
- **Process**:
  - Normalizes queries (lowercase, strip punctuation)
  - Checks cosine similarity against cached queries (threshold: 0.95)
  - Reuses embedding if similarity >= threshold
  - Caches new embeddings (max 1000 entries)
- **Benefits**: Eliminates repeated embedding cost for FAQ queries, donation methods, hospital locations

#### Agentic Workflow

```
User Query
   ↓
Router Agent
   ├─ No Retrieval → Cached/Direct Answer (skip embedding)
   └─ Retrieval Needed
        ↓
Embedding Cache Check
   ├─ Cache Hit → Use cached embedding
   └─ Cache Miss → Generate & cache embedding
        ↓
Retriever Agent
   ├─ Initial Retrieval
   ├─ Relevance Check
   ├─ Low Relevance? → Reformulate Query → Retry Retrieval
   └─ High Relevance → Continue
        ↓
Answer Generation (Self-RAG)
        ↓
Evidence Coverage Agent
   ├─ Split answer into claims
   ├─ Verify each claim against context
   ├─ Remove unsupported claims
   └─ Reject if coverage too low
        ↓
Conversation Memory Agent
   └─ Update state (domain, docs, confidence)
        ↓
Return Answer
```

#### Configuration

**Agentic Features** (in `RAG_supabase.py`):
- `AGENTIC_ENABLE = True` - Enable all agentic workflows
- `EMBEDDING_CACHE_ENABLE = True` - Enable query embedding cache
- `QUERY_ROUTER_ENABLE = True` - Enable query routing before embeddings
- `RETRIEVAL_RETRY_ENABLE = True` - Enable retrieval retry with reformulation
- `EVIDENCE_COVERAGE_ENABLE = True` - Enable claim-by-claim evidence checking
- `CONVERSATION_MEMORY_ENABLE = True` - Enable conversation state reuse

**Thresholds**:
- `EMBEDDING_CACHE_SIMILARITY_THRESHOLD = 0.95` - Reuse embedding if similarity >= this
- `DOMAIN_CENTROID_REUSE_THRESHOLD = 0.85` - Reuse domain embedding if similarity >= this
- `RETRIEVAL_RETRY_MAX_ATTEMPTS = 2` - Maximum retrieval retry attempts
- `RETRIEVAL_RETRY_RELEVANCE_THRESHOLD = 0.6` - Retry if relevance < this

### RAG Components (`RAG_supabase.py`)

#### 1. Self-RAG Implementation
- **Class**: `SelfRAGCritic`
- **Purpose**: Self-reflective RAG system that verifies answer quality at multiple stages
- **Reflection Tokens**: `SelfRAGReflectionTokens` class defines tokens for:
  - `[Retrieve]` / `[No Retrieval]` - Whether retrieval is needed
  - `[Relevant]` / `[Irrelevant]` - Document relevance assessment
  - `[Fully supported]` / `[Partially supported]` / `[No support]` - Answer support verification
  - `[Utility:1-5]` - Answer utility rating (1-5 scale)
- **Critic Methods**:
  1. `is_domain_relevant()` - Checks if query is about Alkhidmat Foundation (rejects irrelevant queries)
  2. `should_retrieve()` - Determines if external knowledge retrieval is needed
  3. `assess_relevance()` - Assesses if retrieved documents are relevant to the query
  4. `check_answer_in_context()` - Verifies if answer can be found in retrieved context
  5. `verify_support()` - Verifies if generated answer is supported by context
  6. `evaluate_utility()` - Evaluates answer usefulness (1-5 scale)
- **Configuration**:
  - `SELFRAG_ENABLE = True` - Enable/disable Self-RAG
  - `SELFRAG_RETRIEVE_THRESHOLD = 0.5` - Threshold for retrieval necessity
  - `SELFRAG_RELEVANCE_THRESHOLD = 0.6` - Threshold for document relevance
  - `SELFRAG_SUPPORT_THRESHOLD = 0.7` - Threshold for answer support
  - `SELFRAG_MIN_CONFIDENCE = 0.6` - Minimum combined confidence to accept answer

#### 2. Enhanced Multilingual Support
- **Class**: `QueryLangProfile`
- **Purpose**: Maintains consistent language handling throughout the pipeline
- **Properties**:
  - `original_query` - What user typed
  - `input_lang` - Detected language: 'en' | 'ur' | 'roman_ur'
  - `query_en` - English version used for embeddings/retrieval/classification
  - `output_lang` - Same as input_lang (answer must match)
  - `query_urdu_script` - Urdu script version (for Roman Urdu queries)
- **Language Detection**:
  - `is_urdu_script()` - Detects Urdu script (Arabic block)
  - `looks_like_roman_urdu()` - Heuristic detection using Roman Urdu markers
  - `detect_language()` - Uses langdetect library
- **Translation Functions**:
  - `translate_urdu_to_english()` - Urdu → English
  - `translate_english_to_urdu()` - English → Urdu (with timeout protection)
  - `translate_auto_to_english()` - Auto-detect → English (for Roman Urdu)
  - `romanize_to_roman_urdu_with_llm()` - Urdu script → Roman Urdu (using LLM)
- **Brand Term Protection**:
  - `protect_brand_terms()` - Protects brand names during translation (e.g., "Alkhidmat", "JazzCash")
  - `restore_brand_terms()` - Restores brand names after translation
  - Prevents corruption of proper nouns during translation

#### 3. Query Analysis & Expansion
- **Function**: `analyze_query(query)`
- **Purpose**: Detects what kind of answer the user wants
- **Returns**:
  - `wants_list` - User wants bullet points/steps
  - `wants_summary` - User wants brief summary
  - `wants_detail` - User wants detailed explanation
  - `is_urdu` - Whether query is in Urdu
- **Function**: `expand_query_for_retrieval(query_en, domain, query_info)`
- **Purpose**: Expands short/procedural queries to improve retrieval
- **Process**:
  - For donation domain: Adds keywords like "donate", "JazzCash", "EasyPaisa", "bank transfer"
  - For healthcare domain: Adds keywords like "hospital", "clinic", "services", "eligibility"
  - Only expands if query is short (< 6 words) or user wants list format

#### 4. Domain Classification
- **Class**: `DomainClassifier`
- **Purpose**: Classify queries into domains (donation, healthcare, general, irrelevant)
- **Method**: 
  - Uses pre-computed domain centroids from anchor queries
  - Calculates cosine similarity between query embedding and domain centroids
  - Returns domain with highest similarity
  - **Enhanced**: Now includes 'irrelevant' domain for queries not about Alkhidmat Foundation

#### 5. Confidence Scoring
- **Class**: `ConfidenceScorer`
- **Purpose**: Calculate confidence scores for RAG responses
- **Metrics**:
  - Retrieval Confidence (cosine similarity with documents)
  - Average Token Confidence (from LLM log probabilities)
  - Weighted Top-K Confidence
  - Perplexity
  - Entropy Confidence
  - **Self-RAG Support Score** (25% weight) - How well answer is supported by context
  - **Self-RAG Utility Score** (15% weight) - Answer usefulness rating
  - **Self-RAG Relevance Score** - Average document relevance
- **Combined Score**: Weighted average:
  - 30% Retrieval Confidence
  - 15% Average Token Confidence
  - 15% Weighted Top-K
  - 25% Self-RAG Support Score
  - 15% Self-RAG Utility Score

#### 6. Document Retrieval
- **Function**: `retrieve_from_supabase(query, top_k=5)`
- **Process**:
  1. Encode query with embedding model (`intfloat/multilingual-e5-base`)
  2. **Query Expansion**: Expand query if needed for better retrieval
  3. Call Supabase RPC function `match_documents` for vector search
  4. Fetch document embeddings for confidence scoring
  5. Return top-k relevant chunks with similarity scores
- **Note**: Always uses English query for retrieval (docs are English-only)

#### 7. Answer Generation (Standard RAG)
- **Function**: `generate_answer(query, top_k=5)`
- **Process**:
  1. **Language Profile**: Build `QueryLangProfile` for multilingual handling
  2. **Query Analysis**: Analyze query to determine answer format preferences
  3. **Domain Classification**: Classify query domain (using English query)
  4. **Query Expansion**: Expand query for better retrieval
  5. **Retrieval**: Get relevant documents from Supabase (using English query)
  6. **Context Building**: Combine retrieved chunks (translate to Urdu if output is Urdu)
  7. **Prompt Construction**: Build prompt with context and query (language-aware)
  8. **LLM Generation**: Generate answer using Llama-cpp-python
  9. **Response Cleaning**: Remove unwanted artifacts from LLM output
  10. **Confidence Calculation**: Calculate all confidence metrics
  11. **Post-processing**: Translate/transliterate answer based on output language
  12. **Return**: Answer, original_query, input_lang, sources, confidence_scores, domain_classification

#### 8. Answer Generation (Self-RAG)
- **Function**: `generate_answer_selfrag(query, top_k=5)`
- **Process**:
  1. **Language Profile**: Build `QueryLangProfile` for multilingual handling
  2. **Query Analysis**: Analyze query to determine answer format preferences
  3. **Step 0 - Domain Relevance**: Check if query is about Alkhidmat Foundation
     - If irrelevant (confidence ≥ 0.75): Reject query, return irrelevant response
  4. **Step 1 - Retrieval Necessity**: Determine if retrieval is needed
     - If not needed: Return message indicating knowledge base access required
  5. **Domain Classification**: Classify query domain (using English query)
  6. **Query Expansion**: Expand query for better retrieval
  7. **Step 2 - Document Retrieval**: Get relevant documents from Supabase
  8. **Step 3 - Relevance Assessment**: Assess each document's relevance
     - Filter out documents below relevance threshold
     - Calculate average relevance score
  9. **Context Building**: Combine relevant chunks (translate if needed)
  10. **Step 3.5 - Answer in Context**: Verify answer exists in context
     - If not found (confidence ≥ 0.7): Return "I don't have that information"
  11. **Prompt Construction**: Build language-aware prompt
  12. **Step 4 - LLM Generation**: Generate answer using Llama-cpp-python
  13. **Response Cleaning**: Remove unwanted artifacts
  14. **Step 5 - Support Verification**: Verify answer is supported by context
     - If no support (confidence ≥ 0.7): Reject answer
  15. **Step 6 - Utility Evaluation**: Evaluate answer usefulness (1-5)
     - If utility ≤ 2: Reject answer
  16. **Confidence Calculation**: Calculate combined confidence with Self-RAG metrics
  17. **Final Check**: If combined confidence < threshold: Reject answer
  18. **Post-processing**: Translate/transliterate answer based on output language
  19. **Return**: Answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics

### Human Agent Request Detection

The system uses comprehensive keyword detection to identify when users request to connect with a human agent:

**Detection Method**: `is_human_agent_request(message)` function in `api_server.py`

**Core Keywords**:
- "connect me with human", "connect me with agent"
- "connect with human", "connect with agent"
- "human agent", "talk to human", "talk to agent"
- "speak with human", "speak with agent"
- "chat with human", "chat with agent"
- "human help", "agent help"
- "need human", "need agent", "want human", "want agent"
- "transfer to human", "transfer to agent"

**Pattern Matching**:
- Checks if message starts with: "connect", "i want", "i need", "can i", "let me"
- Combined with presence of: "agent", "human", "representative", "support"

**Urdu Keywords**:
- "human se baat", "agent se baat"
- "human agent se"
- "human ko connect", "agent ko connect"

**Behavior**:
- If detected → Creates ticket immediately (skips RAG processing)
- Sets domain to "general" for better routing
- Checks for existing active ticket to prevent duplicates
- Returns routing message: "I am connecting you with a human agent. They will respond shortly."

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
4. RAG Processing (Self-RAG enabled by default):
   ├─ Check SELFRAG_ENABLE flag
   ├─ If Self-RAG enabled:
   │  ├─ Use generate_answer_selfrag()
   │  ├─ Step 0: Domain relevance check
   │  ├─ Step 1: Retrieval necessity check
   │  ├─ Domain Classification
   │  ├─ Query Expansion
   │  ├─ Step 2: Document Retrieval
   │  ├─ Step 3: Document Relevance Assessment
   │  ├─ Step 3.5: Answer in Context Check
   │  ├─ Step 4: LLM Answer Generation
   │  ├─ Step 5: Support Verification
   │  ├─ Step 6: Utility Evaluation
   │  └─ Final Confidence Check
   └─ If Self-RAG disabled:
      ├─ Use generate_answer()
      ├─ Domain Classification
      ├─ Query Expansion
      ├─ Document Retrieval
      ├─ LLM Answer Generation
      └─ Confidence Calculation
   ↓
5. Check if user already has active ticket (prevents duplicate tickets)
   ↓
6. Check Confidence Threshold (0.70 for auto-ticket creation)
   ├─ LOW (< 0.70) AND no active ticket → Create ticket, return routing message
   │  ├─ Broadcast ticket creation to agents via WebSocket
   │  └─ Translate routing message to Urdu if user query was in Urdu
   └─ HIGH (>= 0.70) OR has active ticket → Return RAG answer
   ↓
7. Store response in database (with confidence and domain)
   ↓
8. Update query domain in database
   ↓
9. Convert numpy types to native Python types for JSON serialization
   ↓
10. Return response to frontend:
   {
     "answer": "...",
     "sources": [...],
     "agent_chat": false,
     "response_id": "...",
     "confidence": 0.85,
     "ticket_id": null
   }
```

### Self-RAG Metrics Returned

When Self-RAG is enabled, the response includes additional metrics:

```json
{
  "selfrag_metrics": {
    "domain_relevant": true,
    "domain_confidence": 0.85,
    "retrieve_needed": true,
    "retrieve_confidence": 0.80,
    "answer_in_context": true,
    "answer_in_context_confidence": 0.82,
    "relevance_score": 0.75,
    "support_level": "fully_supported",
    "support_score": 0.90,
    "utility_score": 0.80,
    "utility_rating": 4,
    "embedding_cached": false,
    "retrieval_retried": false,
    "evidence_coverage": 1.0,
    "followup_detected": false
  }
}
```

**New Agentic Metrics** (added with Agentic AI):
- `embedding_cached`: Whether query embedding was retrieved from cache (saves ~0.5-1.0s)
- `retrieval_retried`: Whether retrieval was retried with reformulated query (improves relevance)
- `evidence_coverage`: Coverage score (0-1) indicating how many claims are supported by evidence
- `followup_detected`: Whether query was detected as a follow-up to previous conversation

**New Agentic Metrics**:
- `embedding_cached`: Whether query embedding was retrieved from cache
- `retrieval_retried`: Whether retrieval was retried with reformulated query
- `evidence_coverage`: Coverage score (0-1) of claims supported by evidence
- `followup_detected`: Whether query was detected as a follow-up

### Multilingual Processing Flow

```
User Query (any language)
   ↓
Language Detection:
   ├─ Urdu Script → QueryLangProfile(input_lang="ur")
   ├─ Roman Urdu → QueryLangProfile(input_lang="roman_ur")
   └─ English → QueryLangProfile(input_lang="en")
   ↓
Translation to English (for RAG processing):
   ├─ Urdu → English (for embeddings/retrieval)
   ├─ Roman Urdu → English (for embeddings/retrieval)
   └─ English → English (no translation needed)
   ↓
RAG Processing (always uses English):
   ├─ Domain Classification (English query)
   ├─ Query Expansion (English query)
   ├─ Document Retrieval (English query)
   └─ Self-RAG Verification (English query)
   ↓
Answer Generation:
   ├─ Context translation (if output_lang is Urdu/Roman Urdu)
   ├─ Prompt in target language
   └─ LLM generates answer
   ↓
Post-processing:
   ├─ English → No change
   ├─ Urdu → Translate English answer to Urdu
   └─ Roman Urdu → Translate to Urdu, then transliterate to Roman Urdu
   ↓
Return answer in user's original language
```

---

## Enhanced Features

### Self-RAG (Self-Reflective RAG)

The system now includes a Self-RAG implementation based on the research paper "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2024). This adds multiple verification stages to improve answer quality and reliability.

#### Key Benefits:
- **Domain Relevance Filtering**: Rejects queries not about Alkhidmat Foundation
- **Retrieval Necessity Check**: Determines if external knowledge is needed
- **Document Relevance Assessment**: Filters out irrelevant retrieved documents
- **Answer Support Verification**: Ensures answers are supported by context
- **Utility Evaluation**: Rates answer usefulness on a 1-5 scale
- **Automatic Answer Rejection**: Rejects low-quality or unsupported answers

#### Configuration:
Self-RAG can be enabled/disabled via the `SELFRAG_ENABLE` flag in `RAG_supabase.py`. When enabled, the system uses `generate_answer_selfrag()` instead of `generate_answer()`.

#### Answer Rejection Criteria:
An answer is rejected if:
1. Query is irrelevant to Alkhidmat Foundation (domain confidence ≥ 0.75)
2. Retrieved documents are not relevant (relevance < 0.6 threshold)
3. Answer cannot be found in context (confidence ≥ 0.7)
4. Answer is not supported by context (support confidence ≥ 0.7)
5. Answer utility is too low (utility rating ≤ 2)
6. Combined confidence is below minimum threshold (< 0.6)

### Enhanced Multilingual Support

#### Supported Languages:
1. **English** - Native language, no translation needed
2. **Urdu Script** - Arabic script used for Urdu
3. **Roman Urdu** - Urdu written in Latin letters (e.g., "kaise ho", "kya hai")

#### Language Detection:
- **Urdu Script**: Detected via Unicode range `\u0600-\u06FF` (Arabic block)
- **Roman Urdu**: Heuristic detection using marker words:
  - Common words: "kya", "kyu", "kaise", "mein", "aap", "tum"
  - Verbs: "kar", "karo", "hona", "hoga"
  - Particles: "nahi", "han", "hai", "hain"
- **English**: Default fallback

#### Processing Pipeline:
1. **Input Detection**: Detects language from user query
2. **Translation to English**: Converts Urdu/Roman Urdu to English for RAG processing
   - All embeddings, retrieval, and classification use English
   - Ensures consistent performance across languages
3. **Context Translation** (Optional): Can translate context to Urdu before generation
   - Controlled by `TRANSLATE_CONTEXT_FOR_URDU_OUTPUT` flag
   - Improves LLM performance for Urdu output
4. **Answer Translation**: Translates answer back to user's language
   - Urdu script: Direct translation
   - Roman Urdu: Translate to Urdu script, then transliterate to Roman Urdu

#### Brand Term Protection:
During translation, brand terms are protected to prevent corruption:
- Protected terms: "Alkhidmat", "Alkhidmat Foundation", "JazzCash", "EasyPaisa", "Bank of Punjab", "Taqwa Islamic Banking"
- Process: Terms are wrapped with markers (`@@term@@`) before translation, then restored after

### Query Expansion

Short or procedural queries are automatically expanded to improve retrieval:

**Donation Domain:**
- Adds: "donate donation methods how to donate steps JazzCash EasyPaisa bank transfer online donation international account"

**Healthcare Domain:**
- Adds: "Alkhidmat hospital clinic services eligibility locations how to get treatment"

**Expansion Criteria:**
- Query has < 6 words AND user doesn't want list format
- OR user explicitly wants list/steps format

### Enhanced Confidence Scoring

The confidence scoring system now includes Self-RAG metrics:

**Traditional Metrics:**
- Retrieval Confidence (30% weight)
- Average Token Confidence (15% weight)
- Weighted Top-K Confidence (15% weight)

**Self-RAG Metrics:**
- Support Score (25% weight) - How well answer is supported by context
- Utility Score (15% weight) - Answer usefulness rating (normalized 0-1)

**Combined Confidence Formula:**
```
combined = (0.30 × retrieval) + 
           (0.15 × token) + 
           (0.15 × top_k) + 
           (0.25 × support) + 
           (0.15 × utility)
```

### API Integration

The API server (`api_server.py`) automatically uses Self-RAG when enabled:

```python
use_selfrag = getattr(rag_module, 'SELFRAG_ENABLE', False)
if use_selfrag:
    result = await asyncio.to_thread(
        rag_module.generate_answer_selfrag, req.message, top_k=5
    )
    answer, _, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics = result
else:
    answer, _, input_lang, sources, confidence_scores, domain_classification = await asyncio.to_thread(
        rag_module.generate_answer, req.message, top_k=5
    )
```

### JSON Serialization Fix

The system includes a helper function `convert_numpy_types()` that recursively converts numpy types (float32, float64, arrays) to native Python types for JSON serialization. This ensures all responses can be properly serialized when returned to the frontend.

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
   ├─ Processes with RAG (Self-RAG if enabled):
   │  ├─ Domain relevance check (Self-RAG)
   │  ├─ Retrieval necessity check (Self-RAG)
   │  ├─ Domain classification
   │  ├─ Query expansion
   │  ├─ Document retrieval
   │  ├─ Document relevance assessment (Self-RAG)
   │  ├─ Answer in context check (Self-RAG)
   │  ├─ LLM answer generation
   │  ├─ Support verification (Self-RAG)
   │  ├─ Utility evaluation (Self-RAG)
   │  └─ Confidence calculation
   ├─ Checks confidence threshold (0.70)
   ├─ Stores response with confidence and domain
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
   OR types "connect me with agent" (or similar keywords)
   ↓
2. Frontend: API: POST /tickets
   OR
   API: POST /chats/{session_id} (detects agent request via keyword matching)
   ↓
3. Backend: Detects agent request via comprehensive keyword detection
   ├─ Checks for core keywords: "connect me with human/agent", "human agent", etc.
   ├─ Checks for pattern matches: "connect"/"i want"/"i need" + agent keywords
   ├─ Checks Urdu keywords: "human se baat", "agent se baat", etc.
   └─ If match found → Routes immediately (skips RAG)
   ↓
4. Backend: Checks if user already has active ticket
   ├─ If yes → Returns "already connected" message
   └─ If no → Creates ticket
      ├─ Creates placeholder response (domain="general")
      ├─ Creates ticket (status='active')
      ├─ Auto-assigns to domain-matched agent (if available)
      └─ Returns ticket_id
   ↓
5. Frontend: Sets isAgentChat = true
   Shows "connecting to agent" message
   Connects to WebSocket: /ws/user/{session_id}
   Starts polling every 3 seconds (fallback)
   ↓
6. Agent receives ticket in dashboard (via WebSocket or polling)
   ↓
7. Agent assigns ticket → API: POST /tickets/{ticket_id}/assign
   ├─ Updates ticket status to 'in_progress'
   ├─ Sets agent_id
   └─ Broadcasts update to all agents via WebSocket
   ↓
8. Agent sends message → API: POST /tickets/{ticket_id}/message
   ├─ Stores message as response (confidence=1.0)
   ├─ Broadcasts message to user via WebSocket
   └─ Returns success
   ↓
9. User receives message:
   ├─ Via WebSocket (real-time) → Displays immediately
   └─ Via polling (fallback) → API: POST /chats (returns updated chat_history)
   ↓
10. Frontend: Displays agent message
```

### Flow 3: Agent Dashboard Flow

```
1. Agent logs in → API: POST /auth/agent/login
   ├─ Returns agent_token
   └─ Stores token in localStorage
   ↓
2. Redirects to AgentDashboard
   ↓
3. Connects to WebSocket: /ws/agent/{agent_token}
   ├─ Receives real-time ticket updates
   └─ Handles ping/pong for keep-alive
   ↓
4. Loads tickets → API: GET /tickets
   ├─ Filters locally (active, in_progress, resolved)
   ├─ Separates assigned vs unassigned tickets
   └─ Displays in tabs
   ↓
5. Agent selects ticket
   ↓
6. Loads chat → API: GET /tickets/{ticket_id}/chat
   ├─ Checks cache first
   ├─ If cached → Shows immediately
   └─ Fetches from API in background
   ↓
7. Agent types message and sends
   ↓
8. Frontend: Adds message optimistically
   ↓
9. API: POST /tickets/{ticket_id}/message
   ├─ Stores message as response (confidence=1.0)
   ├─ Broadcasts message to user via WebSocket
   └─ Returns success
   ↓
10. Frontend: Refreshes chat after 500ms
    Updates cache
    ↓
11. Agent resolves ticket → API: POST /tickets/{ticket_id}/resolve
    ├─ Updates ticket status to 'resolved'
    ├─ Sets resolved_at timestamp
    ├─ Broadcasts update to all agents via WebSocket
    └─ Frontend: Updates ticket status optimistically
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
│  Backend API    │ 1. Processes with RAG (Self-RAG if enabled)
│  api_server.py  │ 2. Checks confidence threshold (0.70)
│                 │ 3. If low → Creates ticket automatically
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
# Optional: Batch processing mode (suppresses verbose output)
BATCH_MODE=False
```

**Frontend (`.env`):**
```
VITE_API_URL=http://localhost:8000
```

### Confidence Thresholds

**Auto-Ticket Creation Threshold:**
- **Default**: `0.70` (70%)
- **Location**: `api_server.py` → `CONFIDENCE_THRESHOLD = 0.70`
- **Behavior**: If RAG confidence < threshold → Creates ticket automatically, routes to agent
- **Note**: This is separate from Self-RAG minimum confidence threshold

**Self-RAG Minimum Confidence:**
- **Default**: `0.60` (60%)
- **Location**: `RAG_supabase.py` → `SELFRAG_MIN_CONFIDENCE = 0.60`
- **Behavior**: If combined confidence < threshold → Rejects answer, returns "I don't have enough confidence" message
- **Note**: This is stricter than auto-ticket threshold - answers below this are rejected entirely

### Real-Time Communication

**WebSocket Support:**
- **User WebSocket**: `/ws/user/{session_id}` - Receives real-time agent messages
- **Agent WebSocket**: `/ws/agent/{agent_token}` - Receives real-time ticket updates
- **Connection Management**: Automatic reconnection with exponential backoff
- **Message Types**:
  - User receives: `new_message` (from agents)
  - Agent receives: `ticket_update` (created, assigned, resolved)
- **Fallback**: Polling still used as backup if WebSocket fails

**Polling Intervals (Fallback):**
- **User chat (agent messages)**: 3 seconds (only if WebSocket unavailable)
- **Agent dashboard (tickets)**: On mount and visibility change
- **Admin dashboard**: On mount and filter change

### Domain Classification
- **Domains**: `donation`, `healthcare`, `general`, `irrelevant`
- **Method**: Cosine similarity with domain centroids (pre-computed from anchor queries)
- **Location**: `RAG_supabase.py` → `DomainClassifier`
- **Enhanced**: Self-RAG adds domain relevance check that can classify queries as 'irrelevant' if not about Alkhidmat Foundation

### Self-RAG Configuration
- **Enable/Disable**: `SELFRAG_ENABLE = True` in `RAG_supabase.py`
- **Thresholds**:
  - `SELFRAG_RETRIEVE_THRESHOLD = 0.5` - Retrieval necessity confidence
  - `SELFRAG_RELEVANCE_THRESHOLD = 0.6` - Document relevance threshold
  - `SELFRAG_SUPPORT_THRESHOLD = 0.7` - Answer support verification threshold
  - `SELFRAG_MIN_CONFIDENCE = 0.6` - Minimum combined confidence to accept answer
- **Location**: `RAG_supabase.py` (top of file)

### Translation Configuration
- **Timeout**: 15 seconds (configurable in `translate_english_to_urdu()`)
- **Context Translation**: `TRANSLATE_CONTEXT_FOR_URDU_OUTPUT = True` - Translates context to Urdu before generation
- **Brand Terms**: Protected during translation (see `BRAND_TERMS` list in `RAG_supabase.py`)
- **Location**: `RAG_supabase.py` (top of file)

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
   - Session resolution validates UUID format
   - Sessions persist across server restarts
   - Automatic cache restoration from database
2. **Password Hashing**: Agent/admin passwords hashed (currently plaintext for testing)
3. **OTP Storage**: In-memory (use Redis in production)
4. **CORS**: Configured for frontend origin only
5. **Input Validation**: Pydantic models validate all inputs
6. **WebSocket Security**: 
   - User WebSocket validates session_id before connection
   - Agent WebSocket validates agent_token before connection
   - Invalid tokens result in connection rejection (code 1008)

---

## Future Enhancements

1. **Redis Integration**: For OTP storage and session management
2. **Enhanced WebSocket Features**: Bidirectional messaging, typing indicators
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

## Recent Enhancements (2024)

### Self-RAG Integration
- **Added**: Complete Self-RAG implementation with 6 verification stages
- **Benefits**: Improved answer quality, automatic rejection of low-quality answers
- **Status**: Enabled by default (`SELFRAG_ENABLE = True`)

### Enhanced Multilingual Support
- **Added**: Roman Urdu detection and transliteration
- **Added**: `QueryLangProfile` class for consistent language handling
- **Added**: Brand term protection during translation
- **Benefits**: Better support for Urdu-speaking users, preserves proper nouns

### Query Expansion
- **Added**: Automatic query expansion for short/procedural queries
- **Benefits**: Improved retrieval accuracy for donation and healthcare queries

### Enhanced Confidence Scoring
- **Added**: Self-RAG metrics integrated into confidence calculation
- **Updated**: Weight distribution (30% retrieval, 15% token, 15% top-k, 25% support, 15% utility)
- **Benefits**: More accurate confidence scores, better ticket routing decisions

### JSON Serialization Fix
- **Added**: `convert_numpy_types()` helper function
- **Fixed**: TypeError when numpy float32 values were returned in API responses
- **Location**: `api_server.py` and `db_operations.py`

### Code Improvements
- **Updated**: `generate_answer()` now uses multilingual profiles for consistency
- **Updated**: Both standard and Self-RAG paths support Roman Urdu
- **Updated**: Confidence threshold set to 0.70 for auto-ticket creation
- **Added**: WebSocket support for real-time messaging
- **Added**: Comprehensive human agent request detection (keywords + patterns)
- **Added**: Active ticket checking to prevent duplicate tickets
- **Added**: Session resolution helper for robust session management
- **Added**: WebSocket broadcasting for ticket updates (created, assigned, resolved)

### Evidence Coverage Agent Enhancements (2024)
- **Fixed**: Compound claim splitting - Now properly splits sentences like "X provides A, B, and C" into atomic claims
- **Fixed**: Multi-document support - Claims can now be supported by information across multiple documents
- **Added**: Summary mode - Relaxes strictness (70% threshold) for general/about/introduction queries
- **Added**: Smart subject inference - Automatically adds subject to split claims for grammatical correctness
- **Benefits**: 
  - Correctly handles multi-document summaries (e.g., "What is Alkhidmat Foundation?")
  - Prevents false rejections of valid overview answers
  - More accurate claim-by-claim verification for complex answers

### Incremental Knowledge Base Updates (2024)
- **Added**: Incremental document addition - Only processes new documents when updating KB
- **Added**: Document existence checking - Checks if document already exists before processing
- **Added**: Single file addition - `add_single_file_incremental()` for adding individual documents
- **Added**: ZIP incremental mode - `add_documents_from_zip_incremental()` processes only new docs from ZIP
- **Benefits**:
  - **Massive time savings**: Only creates embeddings for new documents (not entire KB)
  - **Cost efficient**: Reduces compute costs for KB updates
  - **Faster updates**: KB can be updated in seconds/minutes instead of hours
  - **Automatic deduplication**: Skips existing documents automatically
- **Usage**:
  - Default mode: `build_alkhidmat_rag(zip_path, incremental=True)` - Only adds new docs
  - Full rebuild: `build_alkhidmat_rag(zip_path, clear_existing=True)` - Rebuilds everything
  - Single file: `add_single_file_incremental(file_path, content, category)` - Add one document

### Multi-Format Document Support (2024)
- **Added**: PDF support - Extracts text from PDF files using PyPDF2
- **Added**: DOCX support - Extracts text from Word documents using python-docx
- **Supported Formats**: 
  - Text files (`.txt`) - UTF-8 and Latin-1 encoding support
  - PDF files (`.pdf`) - Full text extraction from all pages
  - Word documents (`.docx`) - Extracts paragraphs and table content
- **Features**:
  - Automatic format detection based on file extension
  - Graceful error handling for corrupted or unsupported files
  - Table extraction from DOCX files (formatted as pipe-separated values)
  - Multi-page PDF support with page-by-page extraction
- **Dependencies**: 
  - `PyPDF2>=3.0.0` for PDF processing
  - `python-docx>=1.1.0` for DOCX processing
- **Note**: Files are processed from ZIP archives, maintaining the same folder structure (Category/filename.ext)

---

**Last Updated**: December 2024
**Version**: 2.0 (with Self-RAG and Enhanced Multilingual Support)

