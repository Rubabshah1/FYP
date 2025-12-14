#!/usr/bin/env python3
"""
ALKHIDMAT CHAT API - FastAPI Backend (Adapted to Existing Schema)
Uses your existing sessions, queries, responses tables
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

# Import your RAG system
from RAG_supabase import generate_answer, get_supabase_client

app = FastAPI(title="Alkhidmat Chat API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Data Models ============
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict]] = None
    confidence: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # Phone number or UUID
    category: Optional[str] = None
    max_history: int = 5

class ChatResponse(BaseModel):
    session_id: str
    query_id: str
    response_id: str
    message: str
    sources: List[Dict]
    is_urdu: bool
    confidence: Optional[float]
    timestamp: str

# ============ Database Functions ============
def get_or_create_user(phone_number: str) -> str:
    """Get existing user or create new one"""
    supabase = get_supabase_client()
    
    # Check if user exists
    result = supabase.table("users").select("id").eq("phone_number", phone_number).execute()
    
    if result.data:
        return result.data[0]["id"]
    
    # Create new user
    new_user = {
        "phone_number": phone_number,
        "created_at": datetime.now().isoformat()
    }
    
    result = supabase.table("users").insert(new_user).execute()
    return result.data[0]["id"]

def get_or_create_session(session_id: Optional[str], user_id: str) -> str:
    """Get existing session or create new one"""
    supabase = get_supabase_client()
    
    if session_id:
        # Verify session exists and belongs to user
        result = supabase.table("sessions").select("session_id").eq(
            "session_id", session_id
        ).eq("user_id", user_id).execute()
        
        if result.data:
            # Update last_active
            supabase.table("sessions").update({
                "last_active": datetime.now().isoformat()
            }).eq("session_id", session_id).execute()
            
            return session_id
    
    # Create new session
    new_session = {
        "session_id": str(uuid.uuid4()),
        "user_id": user_id,
        "started_at": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
        "metadata": {}
    }
    
    result = supabase.table("sessions").insert(new_session).execute()
    return result.data[0]["session_id"]

def save_query(session_id: str, content: str, domain: Optional[str] = None) -> str:
    """Save user query to database"""
    supabase = get_supabase_client()
    
    query_data = {
        "query_id": str(uuid.uuid4()),
        "session_id": session_id,
        "content": content,
        "domain": domain,
        "timestamp": datetime.now().isoformat()
    }
    
    result = supabase.table("queries").insert(query_data).execute()
    return result.data[0]["query_id"]

def save_response(session_id: str, content: str, confidence: Optional[float], 
                 domain: Optional[str] = None) -> str:
    """Save assistant response to database"""
    supabase = get_supabase_client()
    
    response_data = {
        "response_id": str(uuid.uuid4()),
        "session_id": session_id,
        "content": content,
        "confidence": confidence,
        "domain": domain,
        "timestamp": datetime.now().isoformat()
    }
    
    result = supabase.table("responses").insert(response_data).execute()
    return result.data[0]["response_id"]

def save_response_documents(response_id: str, sources: List[Dict]):
    """Link response to source documents"""
    supabase = get_supabase_client()
    
    if not sources:
        return
    
    response_docs = []
    for i, source in enumerate(sources):
        response_docs.append({
            "id": str(uuid.uuid4()),
            "response_id": response_id,
            "doc_id": source.get("doc_id"),
            "relevance_score": source.get("similarity", 0.0),
            "rank_position": i + 1,
            "created_at": datetime.now().isoformat()
        })
    
    if response_docs:
        supabase.table("response_documents").insert(response_docs).execute()

def get_conversation_history(session_id: str, limit: int = 10) -> List[Dict]:
    """Get conversation history from database"""
    supabase = get_supabase_client()
    
    # Use the stored procedure we created
    try:
        result = supabase.rpc('get_conversation_history', {
            'p_session_id': session_id,
            'p_limit': limit * 2  # Get more because we have queries + responses
        }).execute()
        
        # Convert to our format
        messages = []
        for row in result.data:
            messages.append({
                "role": row["message_type"],
                "content": row["content"],
                "timestamp": row["timestamp"]
            })
        
        # Sort by timestamp
        messages.sort(key=lambda x: x["timestamp"])
        
        return messages[-limit:] if len(messages) > limit else messages
        
    except Exception as e:
        print(f"Error using stored procedure, falling back: {e}")
        
        # Fallback: manual query
        queries = supabase.table("queries").select(
            "content, timestamp"
        ).eq("session_id", session_id).order(
            "timestamp", desc=False
        ).limit(limit).execute()
        
        responses = supabase.table("responses").select(
            "content, timestamp"
        ).eq("session_id", session_id).order(
            "timestamp", desc=False
        ).limit(limit).execute()
        
        # Merge and sort
        messages = []
        for q in queries.data:
            messages.append({
                "role": "user",
                "content": q["content"],
                "timestamp": q["timestamp"]
            })
        
        for r in responses.data:
            messages.append({
                "role": "assistant",
                "content": r["content"],
                "timestamp": r["timestamp"]
            })
        
        messages.sort(key=lambda x: x["timestamp"])
        
        return messages[-limit:] if len(messages) > limit else messages

def build_contextual_query(current_query: str, history: List[Dict]) -> str:
    """Build context-aware query from conversation history"""
    if not history or len(history) < 2:
        return current_query
    
    # Get last 2 exchanges
    recent = history[-4:] if len(history) >= 4 else history
    
    context_parts = []
    for msg in recent:
        if msg["role"] == "user":
            context_parts.append(f"Previous question: {msg['content']}")
        elif msg["role"] == "assistant":
            # Short snippet only
            snippet = msg["content"][:150]
            context_parts.append(f"Previous answer: {snippet}...")
    
    context_parts.append(f"Current question: {current_query}")
    
    return " | ".join(context_parts)

# ============ API Endpoints ============

@app.get("/")
async def root():
    return {
        "service": "Alkhidmat Chat API",
        "version": "2.0",
        "schema": "integrated",
        "endpoints": {
            "chat": "/chat",
            "history": "/history/{session_id}",
            "sessions": "/sessions/{user_id}",
            "sources": "/sources/{response_id}",
            "health": "/health"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - integrated with existing database schema
    """
    try:
        # Get or create user (using phone number or UUID)
        user_id = request.user_id or "anonymous"
        
        if user_id != "anonymous":
            # If phone number provided, get/create user
            if not user_id.startswith('{') and '-' not in user_id:  # Simple check for phone vs UUID
                user_id = get_or_create_user(user_id)
        
        # Get or create session
        session_id = get_or_create_session(request.session_id, user_id)
        
        # Get conversation history
        history = get_conversation_history(session_id, request.max_history)
        
        # Build context-aware query
        enhanced_query = build_contextual_query(request.message, history)
        
        # Save user query to database
        query_id = save_query(session_id, request.message, request.category)
        
        # Call RAG system
        answer, original_query, is_urdu, sources = generate_answer(
            enhanced_query,
            filter_category=request.category
        )
        
        # Calculate confidence (simple heuristic based on source similarity)
        confidence = None
        if sources:
            avg_similarity = sum(s.get("similarity", 0) for s in sources) / len(sources)
            confidence = round(avg_similarity, 3)
        
        # Save response to database
        response_id = save_response(
            session_id, 
            answer, 
            confidence, 
            request.category
        )
        
        # Link response to source documents
        save_response_documents(response_id, sources)
        
        return ChatResponse(
            session_id=session_id,
            query_id=query_id,
            response_id=response_id,
            message=answer,
            sources=sources,
            is_urdu=is_urdu,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    """Get conversation history for a session"""
    try:
        messages = get_conversation_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str, limit: int = 10):
    """Get recent sessions for a user"""
    supabase = get_supabase_client()
    
    try:
        # Use stored procedure if available
        result = supabase.rpc('get_user_recent_sessions', {
            'p_user_id': user_id,
            'p_limit': limit
        }).execute()
        
        return {
            "user_id": user_id,
            "sessions": result.data
        }
    except Exception as e:
        # Fallback
        result = supabase.table("sessions").select(
            "session_id, started_at, last_active"
        ).eq("user_id", user_id).order(
            "last_active", desc=True
        ).limit(limit).execute()
        
        return {
            "user_id": user_id,
            "sessions": result.data
        }

@app.get("/sources/{response_id}")
async def get_sources(response_id: str):
    """Get source documents used in a response"""
    supabase = get_supabase_client()
    
    try:
        # Try stored procedure first
        result = supabase.rpc('get_response_sources', {
            'p_response_id': response_id
        }).execute()
        
        return {
            "response_id": response_id,
            "sources": result.data
        }
    except Exception as e:
        # Fallback
        result = supabase.table("response_documents").select(
            "doc_id, relevance_score, rank_position, documents(category, filename)"
        ).eq("response_id", response_id).order("rank_position").execute()
        
        sources = []
        for row in result.data:
            doc = row.get("documents", {})
            sources.append({
                "doc_id": row["doc_id"],
                "category": doc.get("category"),
                "filename": doc.get("filename"),
                "relevance_score": row["relevance_score"],
                "rank_position": row["rank_position"]
            })
        
        return {
            "response_id": response_id,
            "sources": sources
        }

@app.get("/stats/domains")
async def get_domain_stats():
    """Get statistics by domain"""
    supabase = get_supabase_client()
    
    try:
        # Try to use the view we created
        result = supabase.table("domain_statistics").select("*").execute()
        return {"statistics": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all associated data"""
    supabase = get_supabase_client()
    
    try:
        # Delete cascades to queries and responses due to foreign keys
        supabase.table("sessions").delete().eq("session_id", session_id).execute()
        
        return {
            "message": "Session deleted",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    supabase = get_supabase_client()
    
    try:
        # Test database connection
        result = supabase.table("sessions").select("session_id").limit(1).execute()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============ Run Server ============
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Alkhidmat Chat API Server")
    print("Integrated with existing Supabase schema")
    print("="*60)
    print("\nAPI will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)