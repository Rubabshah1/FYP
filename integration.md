# Chat Integration Guide

## Architecture Overview

```
┌─────────────────┐
│   React App     │  Frontend (Manages UI state)
│  (Port 3000)    │
└────────┬────────┘
         │ HTTP/WebSocket
         ↓
┌─────────────────┐
│  FastAPI Server │  Backend (Manages conversation history)
│  (Port 8000)    │
└────────┬────────┘
         │ Function call
         ↓
┌─────────────────┐
│  RAG System     │  Stateless (Retrieval + Generation)
│ (Python Module) │
└────────┬────────┘
         │ SQL queries
         ↓
┌─────────────────┐
│    Supabase     │  Database (Vector store + Chat history)
│   (Cloud/Self)  │
└─────────────────┘
```

## Setup Steps

### 1. Keep Your RAG System Unchanged ✅
Your `RAG_supabase_client.py` remains **stateless** - no changes needed!

### 2. Install API Dependencies

```bash
pip install fastapi uvicorn
```

### 3. Create the API Layer

Save the `chat_api.py` file in the same directory as your RAG script.

### 4. Start the Backend

```bash
# Terminal 1: Start API server
python chat_api.py
```

The API will run at `http://localhost:8000`

### 5. Test the API

Open http://localhost:8000/docs to see the interactive API documentation.

Try it:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I donate?"}'
```

### 6. Integrate with Frontend

#### Option A: React (Recommended)
Use the `ChatInterface.tsx` component provided in the artifacts.

```bash
# In your React project
npm install
# Make sure API_BASE_URL points to http://localhost:8000
npm start
```

#### Option B: Next.js
Same component works with Next.js - just adjust the import paths.

#### Option C: Plain HTML/JavaScript
```javascript
async function sendMessage(message) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: localStorage.getItem('session_id') })
  });
  
  const data = await response.json();
  localStorage.setItem('session_id', data.session_id);
  return data;
}
```

## How Conversation Context Works

### Before (Single Query)
```
User: "How can I donate?"
RAG: [Searches] → Returns donation methods
```

### After (With Context)
```
User: "How can I donate?"
API stores: { role: 'user', content: 'How can I donate?' }
RAG: [Searches] → Returns donation methods
API stores: { role: 'assistant', content: '...' }

User: "What about online?"  ← Follow-up question!
API enhances query: "Previous: How can I donate? Current: What about online?"
RAG: [Searches with context] → Returns online donation info
```

## Key Features

### 1. Session Management
- Each conversation gets a unique `session_id`
- History stored for 5 previous messages (configurable)
- Sessions automatically created on first message

### 2. Context-Aware Queries
The API automatically enhances queries with conversation history:
```python
# In chat_api.py
def build_contextual_query(current_query, history):
    # Combines previous Q&A with current question
    # Helps RAG understand "what about X?" type questions
```

### 3. Category Filtering
```javascript
// Frontend can filter by category
sendMessage("Show services", sessionId, "Services")
```

### 4. Source Attribution
Sources are passed from RAG → API → Frontend for transparency.

## Optional: Persistent Storage

### Using Memory (Default)
- Sessions stored in `CHAT_SESSIONS` dict
- Lost on server restart
- Good for development

### Using Supabase (Production)
1. Run the SQL schema (`supabase_chat_sessions.sql`)
2. Uncomment the persistence functions in `chat_api.py`
3. Sessions saved to database
4. Survives server restarts

```python
# In chat_api.py, after generating answer:
await save_session_to_db(session_id)
```

## Environment Variables

Create `.env` file:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
GPT4ALL_MODEL=Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

## Production Deployment

### Backend (FastAPI)
```bash
# Using Gunicorn + Uvicorn
gunicorn chat_api:app -w 4 -k uvicorn.workers.UvicornWorker

# Or Railway/Render/Heroku - they auto-detect FastAPI
```

### Frontend (React)
```bash
npm run build
# Deploy to Vercel/Netlify/etc
```

## Debugging

### Check Active Sessions
```bash
curl http://localhost:8000/sessions
```

### View Conversation History
```bash
curl http://localhost:8000/history/{session_id}
```

### Clear a Session
```bash
curl -X DELETE http://localhost:8000/clear/{session_id}
```

## Common Issues

### CORS Errors
Make sure `allow_origins` in `chat_api.py` includes your frontend URL:
```python
allow_origins=["http://localhost:3000", "https://your-domain.com"]
```

### Session Not Persisting
- Check if `session_id` is being stored in frontend (localStorage)
- Verify API returns same `session_id` on subsequent requests

### Context Not Working
- Verify `max_history` > 0 in request
- Check that `build_contextual_query()` is being called
- Look at enhanced query in API logs

## Next Steps

1. **Authentication**: Add user authentication (Supabase Auth)
2. **Rate Limiting**: Prevent abuse
3. **WebSockets**: Real-time streaming responses
4. **Analytics**: Track popular questions
5. **Feedback**: Let users rate answers

## Summary

✅ **RAG stays stateless** - easy to maintain  
✅ **API manages state** - handles conversation history  
✅ **Frontend focuses on UI** - clean separation  
✅ **Scalable architecture** - each layer independent  

# Chat Integration with Your Existing Schema

## Overview

Your existing Supabase schema is **already perfect** for chat! No need for separate chat_sessions table.

### Your Existing Tables (Already Ready!)
- ✅ **users**: Store user info (phone_number)
- ✅ **sessions**: Track conversations (session_id, user_id)
- ✅ **queries**: Store user messages
- ✅ **responses**: Store AI replies
- ✅ **response_documents**: Link responses to source docs
- ✅ **documents**: Your RAG knowledge base

## Setup Steps

### 1. Run SQL Setup in Supabase

Go to Supabase SQL Editor and run the SQL from the artifact. This adds:
- Helper functions for getting conversation history
- Indexes for better performance
- Optional analytics views
- Row Level Security (RLS) policies

```sql
-- Copy and paste the entire supabase_chat_sessions.sql content
-- from the artifact into Supabase SQL Editor and run it
```

### 2. Update Your FastAPI

Replace your `chat_api.py` with the new version that uses your existing schema.

### 3. Test the Integration

```bash
# Start the API
python chat_api.py

# Test with curl
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I donate?",
    "user_id": "+923001234567"
  }'
```

## How It Works

### First Message (New User)
```
1. User sends: "How can I donate?" with phone "+923001234567"
2. API creates new user in users table
3. API creates new session in sessions table
4. API saves query to queries table
5. RAG generates answer
6. API saves response to responses table
7. API links sources to response_documents table
```

### Follow-up Message (Existing Session)
```
1. User sends: "What about online?" with session_id from previous response
2. API retrieves conversation history from queries + responses
3. API enhances query: "Previous: How can I donate? Current: What about online?"
4. RAG generates context-aware answer
5. API saves new query and response
```

### Database After 2 Messages

**users table:**
```
id                                    | phone_number
--------------------------------------|-------------
550e8400-e29b-41d4-a716-446655440000 | +923001234567
```

**sessions table:**
```
session_id                            | user_id                              | started_at | last_active
--------------------------------------|--------------------------------------|------------|------------
660e8400-e29b-41d4-a716-446655440001 | 550e8400-e29b-41d4-a716-446655440000 | 2024-...   | 2024-...
```

**queries table:**
```
query_id | session_id | content                  | timestamp
---------|------------|--------------------------|----------
...001   | ...001     | How can I donate?        | 2024-...
...002   | ...001     | What about online?       | 2024-...
```

**responses table:**
```
response_id | session_id | content                       | confidence | timestamp
------------|------------|-------------------------------|------------|----------
...101      | ...001     | You can donate via...         | 0.857      | 2024-...
...102      | ...001     | For online donations, visit...| 0.849      | 2024-...
```

**response_documents table:**
```
id   | response_id | doc_id | relevance_score | rank_position
-----|-------------|--------|-----------------|---------------
...  | ...101      | ...    | 0.865          | 1
...  | ...101      | ...    | 0.857          | 2
```

## API Endpoints

### 1. Send Message
```bash
POST /chat
{
  "message": "How can I donate?",
  "user_id": "+923001234567",  # Phone number or UUID
  "session_id": "optional-existing-session-id",
  "category": "Donors",  # Optional filter
  "max_history": 5  # How many previous messages to include
}

Response:
{
  "session_id": "uuid",
  "query_id": "uuid",
  "response_id": "uuid",
  "message": "You can donate...",
  "sources": [...],
  "is_urdu": false,
  "confidence": 0.857,
  "timestamp": "2024-..."
}
```

### 2. Get Conversation History
```bash
GET /history/{session_id}?limit=20

Response:
{
  "session_id": "uuid",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "count": 10
}
```

### 3. Get User's Sessions
```bash
GET /sessions/{user_id}?limit=10

Response:
{
  "user_id": "uuid",
  "sessions": [
    {
      "session_id": "uuid",
      "started_at": "...",
      "last_active": "...",
      "message_count": 5,
      "first_message": "How can I donate?"
    }
  ]
}
```

### 4. Get Response Sources
```bash
GET /sources/{response_id}

Response:
{
  "response_id": "uuid",
  "sources": [
    {
      "doc_id": "uuid",
      "category": "Donors",
      "filename": "donate-summary.txt",
      "relevance_score": 0.865,
      "rank_position": 1
    }
  ]
}
```

### 5. Get Domain Statistics
```bash
GET /stats/domains

Response:
{
  "statistics": [
    {
      "domain": "Donors",
      "query_count": 150,
      "unique_sessions": 45,
      "avg_confidence": 0.82
    }
  ]
}
```

## Frontend Integration

### React Example
```typescript
const [sessionId, setSessionId] = useState<string | null>(null);
const [userId] = useState('+923001234567'); // From auth

async function sendMessage(message: string) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      user_id: userId,
      session_id: sessionId,  // null for first message
      max_history: 5
    })
  });
  
  const data = await response.json();
  
  // Save session_id for future messages
  if (!sessionId) {
    setSessionId(data.session_id);
  }
  
  return data;
}
```

### Load Previous Sessions
```typescript
async function loadSessions() {
  const response = await fetch(
    `http://localhost:8000/sessions/${userId}?limit=10`
  );
  const data = await response.json();
  return data.sessions;
}

async function loadHistory(sessionId: string) {
  const response = await fetch(
    `http://localhost:8000/history/${sessionId}?limit=20`
  );
  const data = await response.json();
  return data.messages;
}
```

## Key Features

### ✅ Conversation Context
- Automatically loads last 5 messages
- Enhances queries with context
- Understands follow-up questions

### ✅ Source Tracking
- Every response linked to source documents
- Track which docs are most useful
- Show users where info came from

### ✅ Analytics Ready
- Track popular domains
- Monitor confidence scores
- Analyze user patterns

### ✅ Multi-User Support
- Phone number identification
- User can have multiple sessions
- Session history preserved

### ✅ Security
- Row Level Security (RLS) enabled
- Users only see their own data
- Backend can access all sessions

## Database Queries You Can Run

### Get user's most recent conversation
```sql
SELECT * FROM get_conversation_history('session-id-here', 10);
```

### Get user's all sessions
```sql
SELECT * FROM get_user_recent_sessions('user-id-here', 10);
```

### Find most popular topics
```sql
SELECT * FROM domain_statistics;
```

### Get session details with stats
```sql
SELECT * FROM session_stats WHERE session_id = 'session-id-here';
```

## Optional Cleanup

### Archive old sessions (run monthly)
```sql
SELECT archive_old_sessions(90); -- Keep last 90 days
```

### Manual cleanup
```sql
-- Delete sessions older than 6 months
DELETE FROM sessions 
WHERE last_active < NOW() - INTERVAL '6 months';
```

## Advantages of This Approach

1. **No Duplicate Data**: Uses existing schema, no redundancy
2. **Proper Relationships**: Foreign keys ensure data integrity
3. **Source Attribution**: Every answer linked to documents used
4. **Analytics Built-in**: Track everything without extra tables
5. **Scalable**: Designed for production use
6. **Secure**: RLS policies protect user data

## Next Steps

1. ✅ Run SQL setup in Supabase
2. ✅ Update FastAPI code
3. ✅ Test endpoints
4. ✅ Integrate with frontend
5. 🔄 Monitor performance
6. 📊 Review analytics

Your database is now ready for production chat! 🚀