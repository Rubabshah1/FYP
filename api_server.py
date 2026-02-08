"""
FastAPI wrapper to serve the Alkhidmat RAG pipeline using RAG_supabase.py.
- Authentication endpoints for users (OTP), agents, and admins
- Chat endpoints with session management
- Ticket management endpoints
- Analytics endpoints for admins

Environment variables:
- SUPABASE_URL         : Supabase project URL
- SUPABASE_KEY         : Supabase anon key (or SUPABASE_ANON_KEY)
- ALKHIDMAT_ZIP_PATH   : path to the knowledge base zip (default: "Al Khidmat Knowledge Base.zip")
                          Only needed for initial KB build
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
from datetime import datetime, timedelta
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from supabase_client import init_supabase
from db_operations import (
    create_user, get_user_by_phone, get_user_by_id, generate_otp, verify_otp,
    create_session, get_session, update_session_activity,
    create_query, update_query_domain, create_response, create_ticket, get_ticket, list_tickets,
    assign_ticket, resolve_ticket, authenticate_agent, authenticate_admin,
    get_ticket_analytics, get_agent, get_admin,
    get_user_chat_history, get_session_chat_history
)

# Import RAG_supabase module
import RAG_supabase as rag_module

ZIP_PATH = os.getenv("ALKHIDMAT_ZIP_PATH", "Al Khidmat Knowledge Base.zip")

app = FastAPI(title="Alkhidmat RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (in production, use Redis or JWT tokens)
user_sessions: dict = {}  # {session_token: {"user_id": user_id, "db_session_id": db_session_id}}
agent_sessions: dict = {}  # {token: agent_id}
admin_sessions: dict = {}  # {token: admin_id}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalar to Python native type
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        # {session_id: [websocket1, websocket2, ...]} for users
        # {agent_id: [websocket1, websocket2, ...]} for agents
        self.user_connections: dict = {}
        self.agent_connections: dict = {}
    
    async def connect_user(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.user_connections:
            self.user_connections[session_id] = []
        self.user_connections[session_id].append(websocket)
        print(f"[WS] User connected: {session_id} (total: {len(self.user_connections[session_id])})")
    
    def disconnect_user(self, websocket: WebSocket, session_id: str):
        if session_id in self.user_connections:
            if websocket in self.user_connections[session_id]:
                self.user_connections[session_id].remove(websocket)
            if len(self.user_connections[session_id]) == 0:
                del self.user_connections[session_id]
        print(f"[WS] User disconnected: {session_id}")
    
    async def send_to_user(self, session_id: str, message: dict):
        if session_id in self.user_connections:
            disconnected = []
            for websocket in self.user_connections[session_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    print(f"[WS] Error sending to user {session_id}: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.user_connections[session_id].remove(ws)
    
    async def connect_agent(self, websocket: WebSocket, agent_id: str):
        await websocket.accept()
        if agent_id not in self.agent_connections:
            self.agent_connections[agent_id] = []
        self.agent_connections[agent_id].append(websocket)
        print(f"[WS] Agent connected: {agent_id} (total: {len(self.agent_connections[agent_id])})")
    
    def disconnect_agent(self, websocket: WebSocket, agent_id: str):
        if agent_id in self.agent_connections:
            if websocket in self.agent_connections[agent_id]:
                self.agent_connections[agent_id].remove(websocket)
            if len(self.agent_connections[agent_id]) == 0:
                del self.agent_connections[agent_id]
        print(f"[WS] Agent disconnected: {agent_id}")
    
    async def send_to_agent(self, agent_id: str, message: dict):
        if agent_id in self.agent_connections:
            disconnected = []
            for websocket in self.agent_connections[agent_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    print(f"[WS] Error sending to agent {agent_id}: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.agent_connections[agent_id].remove(ws)
    
    async def broadcast_ticket_update(self, ticket_id: str, update_type: str, data: dict):
        """Broadcast ticket updates to all connected agents"""
        message = {
            "type": "ticket_update",
            "ticket_id": ticket_id,
            "update_type": update_type,
            "data": data
        }
        # Broadcast to all agents
        for agent_id, connections in list(self.agent_connections.items()):
            for websocket in connections:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    print(f"[WS] Error broadcasting to agent {agent_id}: {e}")

manager = ConnectionManager()

# ============================================================================
# REQUEST MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str


class OTPRequest(BaseModel):
    phone_number: str


class OTPVerifyRequest(BaseModel):
    phone_number: str
    otp: str


class AgentLoginRequest(BaseModel):
    email: str
    password: str


class AdminLoginRequest(BaseModel):
    email: str
    password: str


class TicketRequest(BaseModel):
    session_id: str
    initial_message: str = ""


class MessageRequest(BaseModel):
    message: str
    sender: str = "agent"  # "agent" or "user"


# ============================================================================
# AUTHENTICATION HELPERS
# ============================================================================

def get_current_user(session_id: str = Header(..., alias="X-Session-ID")):
    """Get current user from session."""
    if session_id not in user_sessions:
        raise HTTPException(status_code=401, detail="Invalid session")
    session_data = user_sessions[session_id]
    user_id = session_data.get("user_id") if isinstance(session_data, dict) else session_data
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_agent(token: str = Header(..., alias="X-Agent-Token")):
    """Get current agent from token."""
    if token not in agent_sessions:
        raise HTTPException(status_code=401, detail="Invalid agent token")
    agent_id = agent_sessions[token]
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=401, detail="Agent not found")
    return agent


def get_current_admin(token: str = Header(..., alias="X-Admin-Token")):
    """Get current admin from token."""
    if token not in admin_sessions:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    admin_id = admin_sessions[token]
    admin = get_admin(admin_id)
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found")
    return admin


# ============================================================================
# STARTUP
# ============================================================================

def _should_check_for_updates() -> bool:
    """
    Check if it's time to check for KB updates (weekly basis).
    Returns True if a week has passed since last check, False otherwise.
    """
    timestamp_file = Path(".kb_last_check.json")
    check_interval_days = 7  # Weekly check
    
    if not timestamp_file.exists():
        # First time - create timestamp file and return True
        timestamp_file.write_text(json.dumps({
            "last_check": datetime.now().isoformat()
        }))
        return True
    
    try:
        data = json.loads(timestamp_file.read_text())
        last_check_str = data.get("last_check")
        if not last_check_str:
            return True
        
        last_check = datetime.fromisoformat(last_check_str)
        days_since_check = (datetime.now() - last_check).days
        
        if days_since_check >= check_interval_days:
            # Update timestamp
            timestamp_file.write_text(json.dumps({
                "last_check": datetime.now().isoformat()
            }))
            return True
        else:
            days_remaining = check_interval_days - days_since_check
            print(f"  KB update check skipped (last checked {days_since_check} day(s) ago, next check in {days_remaining} day(s))")
            return False
    except Exception as e:
        print(f"  ⚠️ Error reading KB check timestamp: {e}")
        # On error, allow check to proceed
        return True

def _ensure_index_exists():
    """Check if knowledge base exists in Supabase, build if missing, update incrementally if exists."""
    from supabase_client import get_supabase_client
    
    # Check if ZIP file exists
    if not Path(ZIP_PATH).exists():
        print(f"  ZIP file not found at {ZIP_PATH}")
        print("   Please set ALKHIDMAT_ZIP_PATH environment variable or place the ZIP file in the expected location.")
        print("   The server will start, but RAG queries will fail until the knowledge base is built.")
        return
    
    try:
        # Check if documents exist in Supabase
        supabase = get_supabase_client()
        if supabase:
            result = supabase.table("documents").select("doc_id", count="exact").limit(1).execute()
            doc_count = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
            
            if doc_count > 0:
                print(f"✓ Knowledge base found in Supabase ({doc_count} documents).")
                
                # Only check for updates on a weekly basis
                if _should_check_for_updates():
                    print(f"  Checking for new documents in ZIP file (weekly check)...")
                    # Run incremental update to add any new documents
                    try:
                        stats = rag_module.add_documents_from_zip_incremental(ZIP_PATH, reindex_existing=False)
                        if stats["added"] > 0:
                            print(f"✓ Added {stats['added']} new document(s) to knowledge base")
                        if stats["skipped"] > 0:
                            print(f"  Skipped {stats['skipped']} existing document(s)")
                        if stats["failed"] > 0:
                            print(f"⚠️ Failed to add {stats['failed']} document(s)")
                    except Exception as e:
                        print(f"⚠️ Error during incremental update: {e}")
                        print("   Knowledge base exists, but new documents may not have been added.")
                else:
                    print(f"  KB update check skipped (weekly schedule)")
                return
        else:
            print("  Supabase client not initialized. Skipping knowledge base check.")
            return
    except Exception as e:
        print(f"  Could not check Supabase documents: {e}")
        print("   The server will start, but RAG queries may fail if the knowledge base is not built.")
        return
    
    # No documents found, need to build from scratch
    print(f"  Knowledge base not found in Supabase. Building now...")
    try:
        # Build from scratch (not incremental since there's nothing to increment)
        rag_module.build_alkhidmat_rag(ZIP_PATH, clear_existing=False, incremental=False)
        print(f"✓ Knowledge base built successfully in Supabase")
        # Update timestamp after initial build
        timestamp_file = Path(".kb_last_check.json")
        timestamp_file.write_text(json.dumps({
            "last_check": datetime.now().isoformat()
        }))
    except Exception as e:
        print(f"  Error building knowledge base: {e}")
        print("   The server will start, but RAG queries will fail until the knowledge base is built.")


@app.on_event("startup")
async def _startup():
    # Initialize Supabase client (non-blocking - will warn if not configured)
    init_supabase()
    # Build index in a thread to avoid blocking the event loop on startup.
    await asyncio.to_thread(_ensure_index_exists)
    
    # Pre-initialize domain embeddings to avoid delay on first query
    print("[STARTUP] Pre-initializing domain embeddings...", flush=True)
    try:
        from RAG_supabase import DomainClassifier
        DomainClassifier.initialize_domain_embeddings()
        print("[STARTUP] ✓ Domain embeddings ready", flush=True)
    except Exception as e:
        print(f"[STARTUP]   Could not pre-initialize domain embeddings: {e}", flush=True)
    
    # Pre-load embedding model to avoid delay on first query
    print("[STARTUP] Pre-loading embedding model...", flush=True)
    try:
        from RAG_supabase import get_embedder
        get_embedder()
        print("[STARTUP] ✓ Embedding model ready", flush=True)
    except Exception as e:
        print(f"[STARTUP]   Could not pre-load embedding model: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================================
# USER AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/user/send-otp")
async def send_otp(req: OTPRequest):
    """Send OTP to phone number."""
    otp = generate_otp(req.phone_number)
    # In production, send OTP via SMS service
    print(f"OTP for {req.phone_number}: {otp}")  # For development only
    return {"message": "OTP sent", "otp": otp}  # Remove otp in production


@app.post("/auth/user/verify-otp")
async def verify_otp_endpoint(req: OTPVerifyRequest):
    """Verify OTP and create/login user."""
    if not verify_otp(req.phone_number, req.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    # Get or create user
    user = get_user_by_phone(req.phone_number)
    if not user:
        user = create_user(req.phone_number)
    
    # Create new session for this login
    session = create_session(user["id"])
    db_session_id = str(session["session_id"])
    
    # Use database session_id as the session token (it's a UUID)
    # This way it persists across server restarts
    session_token = db_session_id
    
    # Store in memory for quick lookup
    user_sessions[session_token] = {
        "user_id": user["id"],
        "db_session_id": session["session_id"]
    }
    
    # Get all chat history for this user (across all sessions) - WhatsApp-like behavior
    chat_history = get_user_chat_history(user["id"])
    
    return {
        "session_id": session_token,  # This is now the database session_id (UUID)
        "user_id": str(user["id"]),
        "session": {
            "session_id": str(session["session_id"]),
            "started_at": session.get("started_at") or None
        },
        "chat_history": chat_history  # Include all previous messages
    }


# ============================================================================
# AGENT AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/agent/login")
async def agent_login(req: AgentLoginRequest):
    """Login agent."""
    agent = authenticate_agent(req.email, req.password)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    import secrets
    token = secrets.token_urlsafe(32)
    agent_sessions[token] = agent["agent_id"]
    
    return {
        "token": token,
        "agent_id": str(agent["agent_id"]),
        "name": f"{agent['first_name']} {agent['last_name']}",
        "email": agent["email"],
        "domain": agent.get("domain")
    }


# ============================================================================
# ADMIN AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/admin/login")
async def admin_login(req: AdminLoginRequest):
    """Login admin."""
    admin = authenticate_admin(req.email, req.password)
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    import secrets
    token = secrets.token_urlsafe(32)
    admin_sessions[token] = admin["admin_id"]
    
    return {
        "token": token,
        "admin_id": str(admin["admin_id"]),
        "name": f"{admin['first_name']} {admin['last_name']}",
        "email": admin["email"]
    }


# ============================================================================
# SESSION RESOLUTION HELPER
# ============================================================================

def resolve_session_id(session_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a session_id (which could be an old token or a UUID) to a database session_id.
    Returns (db_session_id, error_message) tuple.
    - If successful: (db_session_id, None)
    - If failed: (None, error_message)
    """
    if not session_id:
        return None, "Session ID is required"
    
    # Check in-memory cache first
    if session_id in user_sessions:
        session_data = user_sessions[session_id]
        if isinstance(session_data, dict):
            return session_data.get("db_session_id"), None
        return session_id, None
    
    # Check if it's a valid UUID format
    from uuid import UUID
    is_valid_uuid = False
    try:
        UUID(session_id)
        is_valid_uuid = True
    except (ValueError, TypeError):
        pass
    
    if is_valid_uuid:
        # Try to look up in database
        try:
            db_session = get_session(session_id)
            if db_session:
                # Restore to in-memory cache
                user_sessions[session_id] = {
                    "user_id": db_session["user_id"],
                    "db_session_id": db_session["session_id"]
                }
                return db_session["session_id"], None
            else:
                return None, "Session not found in database. Please login again."
        except Exception as e:
            return None, f"Error looking up session: {str(e)}"
    else:
        # Old token format - not stored in database, only in memory (lost on restart)
        return None, "Session expired (old token format). Please login again to get a new session."


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@app.post("/chats")
async def create_chat(session_token: Optional[str] = Header(None, alias="X-Session-ID")):
    """Create a new chat session and return chat history."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Please login first. Session token required in X-Session-ID header.")
    
    db_session_id, error_msg = resolve_session_id(session_token)
    
    if not db_session_id:
        raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
    # Get session details
    db_session = get_session(db_session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = db_session["user_id"]
    
    # Get all chat history for this user (across all sessions)
    chat_history = get_user_chat_history(user_id)
    
    return {
        "session_id": session_token,  # Return the token that was sent (could be UUID or old format)
        "db_session_id": str(db_session_id),
        "user_id": str(user_id),
        "chat_history": chat_history  # Include all previous messages
    }


@app.get("/chats/{session_id}/history")
async def get_chat_history(
    session_id: str,
    session_token: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Get chat history for a session."""
    session_to_resolve = session_token or session_id
    
    db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
    if not db_session_id:
        raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
    # Get chat history for this specific session
    chat_history = get_session_chat_history(db_session_id)
    
    return {
        "session_id": str(db_session_id),
        "messages": chat_history
    }


@app.post("/chats/{session_id}")
async def chat(
    session_id: str, 
    req: ChatRequest,
    session_token: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Process chat message with RAG."""
    # Prefer X-Session-ID header if provided, otherwise use URL parameter
    session_to_resolve = session_token or session_id
    
    # Debug logging
    print(f"[DEBUG] Chat request - URL session_id: {session_id}, Header X-Session-ID: {session_token}, Resolving: {session_to_resolve}")
    
    db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
    if not db_session_id:
        print(f"[DEBUG] Session resolution failed: {error_msg}")
        raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
    # Get or verify session
    db_session = get_session(db_session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update session activity
    update_session_activity(db_session_id)
    
    # Create query record (domain will be updated after RAG processing)
    query_record = create_query(db_session_id, req.message)
    query_id = query_record["query_id"] if query_record else None
    
    # Check if user message requests human agent
    def is_human_agent_request(message: str) -> bool:
        """Detect if user message requests to connect with human agent."""
        if not message:
            return False
        
        # Clean message: remove trailing punctuation and normalize
        message_clean = message.lower().strip()
        # Remove trailing punctuation (., !, ?, ...)
        while message_clean and message_clean[-1] in '.,!?…':
            message_clean = message_clean[:-1].strip()
        
        # Core keywords that indicate agent request
        core_keywords = [
            "connect me with human",
            "connect me with agent",
            "connect with human",
            "connect with agent",
            "connect me to agent",
            "connect me to human",
            "connect to human",
            "connect to agent",
        ]
        
        # Additional keywords
        keywords = core_keywords + [
            "human agent",
            "talk to human",
            "talk to agent",
            "speak with human",
            "speak with agent",
            "chat with human",
            "chat with agent",
            "human help",
            "agent help",
            "need human",
            "need agent",
            "want human",
            "want agent",
            "transfer to human",
            "transfer to agent",
            "i want to talk to",
            "can i speak to",
            "can i talk to",
            "let me talk to",
            "let me speak to",
            "i need to talk to",
            "i need to speak to",
            "i want to connect with agent",
            "i want to connect with human",
            "i need to connect with agent",
            "i need to connect with human",
            "want to talk to agent",
            "want to talk to human",
            "want to speak to agent",
            "want to speak to human",
            # Urdu keywords
            "human se baat",
            "agent se baat",
            "human agent se",
            "human ko connect",
            "agent ko connect",
        ]
        
        # Check for matches
        matched = any(keyword in message_clean for keyword in keywords)
        
        # Also check if message starts with common patterns (more lenient)
        if not matched:
            starts_with_patterns = [
                "connect",
                "i want",
                "i need",
                "can i",
                "let me",
            ]
            contains_agent_keywords = ["agent", "human", "representative", "support"]
            
            if any(message_clean.startswith(pattern) for pattern in starts_with_patterns):
                if any(keyword in message_clean for keyword in contains_agent_keywords):
                    matched = True
        
        if matched:
            print(f"[HUMAN-AGENT-DETECTION] ✅ Detected human agent request in message: '{message}' (cleaned: '{message_clean}')")
        else:
            print(f"[HUMAN-AGENT-DETECTION] ❌ No match for message: '{message}' (cleaned: '{message_clean}')")
        
        return matched
    
    try:
        # Check if user has an active ticket for this session
        from supabase_client import get_supabase_client
        supabase = get_supabase_client()
        # Check for active tickets linked to responses in this session
        active_responses = supabase.table("responses").select("response_id").eq(
            "session_id", db_session_id
        ).execute()
        response_ids = [r["response_id"] for r in (active_responses.data or [])]
        
        has_active_ticket = False
        if response_ids:
            active_tickets = supabase.table("tickets").select("ticket_id").in_(
                "response_id", response_ids
            ).in_("status", ["active", "in_progress"]).execute()
            has_active_ticket = len(active_tickets.data) > 0 if active_tickets.data else False
        
        # Check if user explicitly requested human agent
        user_requested_agent = is_human_agent_request(req.message)
        print(f"[HUMAN-AGENT-CHECK] user_requested_agent={user_requested_agent}, has_active_ticket={has_active_ticket}")
        
        # If user requested agent, route immediately WITHOUT RAG processing
        if user_requested_agent:
            print(f"[HUMAN-AGENT-REQUEST] User requested human agent: '{req.message}' - Routing immediately, skipping RAG")
            
            # If there's already an active ticket, just inform user they're already connected
            if has_active_ticket:
                print(f"[HUMAN-AGENT-REQUEST] User already has active ticket - informing them")
                return {
                    "answer": "You are already connected with a human agent. They will respond shortly.",
                    "sources": [],
                    "agent_chat": True,
                    "response_id": None,
                    "confidence": 1.0,
                    "ticket_id": None
                }
            
            # Set domain to "general" for agent connection requests
            # This ensures tickets are routed to general domain agents
            query_domain = "general"
            print(f"[HUMAN-AGENT-REQUEST] Setting domain to 'general' for agent connection request")
            
            # Update query with general domain
            if query_id:
                update_query_domain(query_id, query_domain)
                print(f"[QUERY-DOMAIN] Updated query {query_id} with domain: {query_domain}")
            
            # Create a placeholder response for the ticket
            placeholder_response = create_response(
                db_session_id,
                f"User requested to chat with human agent: {req.message}",
                confidence=1.0,  # High confidence since it's an explicit request
                domain=query_domain
            )
            # Create ticket immediately with domain for better routing
            ticket = create_ticket(placeholder_response["response_id"], domain=query_domain)
            if ticket:
                ticket_id = str(ticket.get("ticket_id"))
                agent_assigned = ticket.get("agent_id")
                status = ticket.get("status")
                print(f"[HUMAN-AGENT-REQUEST] Created ticket {ticket_id} (status: {status}, domain: {query_domain})")
                if agent_assigned:
                    print(f"[HUMAN-AGENT-REQUEST] Auto-assigned to agent {agent_assigned}")
                
                # Return early with agent chat response - DO NOT PROCESS WITH RAG
                return {
                    "answer": "I am connecting you with a human agent. They will respond shortly.",
                    "sources": [],
                    "agent_chat": True,
                    "response_id": str(placeholder_response["response_id"]),
                    "confidence": 1.0,
                    "ticket_id": ticket_id
                }
            else:
                # If ticket creation failed, still return agent routing message
                print(f"[HUMAN-AGENT-REQUEST] WARNING: Ticket creation failed, but user requested agent")
                return {
                    "answer": "I am connecting you with a human agent. They will respond shortly.",
                    "sources": [],
                    "agent_chat": True,
                    "response_id": str(placeholder_response["response_id"]),
                    "confidence": 1.0,
                    "ticket_id": None
                }
        
        # Process with RAG (use Self-RAG if enabled)
        use_selfrag = getattr(rag_module, 'SELFRAG_ENABLE', False)
        if use_selfrag:
            # generate_answer_selfrag returns (answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics)
            # Pass session_id for conversation memory
            result = await asyncio.to_thread(
                rag_module.generate_answer_selfrag, req.message, top_k=5, filter_category=None, session_id=str(db_session_id)
            )
            answer, _, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics = result
            is_urdu = (input_lang == "ur" or input_lang == "roman_ur")
        else:
            # generate_answer returns (answer, query, is_urdu, sources, confidence_scores, domain_classification)
            answer, _, is_urdu, sources, confidence_scores, domain_classification = await asyncio.to_thread(
                rag_module.generate_answer, req.message, top_k=5, filter_category=None
            )
        
        # Use confidence from RAG module (combined confidence score)
        confidence = None
        if confidence_scores and isinstance(confidence_scores, dict):
            # Use combined confidence if available
            confidence = confidence_scores.get("combined_confidence") or confidence_scores.get("retrieval_confidence")
        elif isinstance(confidence_scores, (int, float)):
            confidence = float(confidence_scores)
        
        # Fallback: Calculate confidence from sources (average similarity score)
        if confidence is None and sources and len(sources) > 0:
            similarities = [s.get("similarity", 0.0) for s in sources if s.get("similarity") is not None]
            if similarities:
                confidence = sum(similarities) / len(similarities)
        
        # Convert confidence to native Python float if it's a numpy type
        if confidence is not None:
            confidence = convert_numpy_types(confidence)
        
        # Use domain classification from RAG module
        query_domain = None
        if domain_classification and isinstance(domain_classification, dict):
            # Get the primary domain (highest confidence)
            if "primary_domain" in domain_classification:
                query_domain = domain_classification["primary_domain"]
            elif "domain" in domain_classification:
                query_domain = domain_classification["domain"]
        elif isinstance(domain_classification, str):
            query_domain = domain_classification
        
        # Fallback: Extract domain from first source (if available)
        if not query_domain and sources and len(sources) > 0:
            if sources[0].get("category"):
                query_domain = sources[0].get("category")
        
        # Log domain classification for debugging
        print(f"[DOMAIN-CLASSIFICATION] Domain classification result: {domain_classification}")
        print(f"[DOMAIN-CLASSIFICATION] Extracted domain: '{query_domain}'")
        
        # Update query record with domain if it was determined
        if query_id and query_domain:
            update_query_domain(query_id, query_domain)
            print(f"[QUERY-DOMAIN] Updated query {query_id} with domain: {query_domain}")
        
        # Create response record with confidence
        response = create_response(
            db_session_id, 
            answer, 
            confidence=confidence,
            domain=query_domain
        )
        
        # Auto-create ticket if confidence is below threshold (0.5-0.6)
        # Note: user_requested_agent is already handled above with early return
        CONFIDENCE_THRESHOLD = 0.70  # Configurable threshold
        ticket_created = False
        ticket_id = None
        final_answer = answer  # Default to original answer
        
        # Create ticket if confidence is low (user-requested agent already handled above)
        should_create_ticket = (
            confidence is not None and confidence < CONFIDENCE_THRESHOLD
        ) and not has_active_ticket
        
        if should_create_ticket:
            # Create ticket automatically with domain routing
            ticket = create_ticket(response["response_id"], domain=query_domain)
            if ticket:
                ticket_created = True
                ticket_id = str(ticket.get("ticket_id"))
                agent_assigned = ticket.get("agent_id")
                status = ticket.get("status")
                print(f"[AUTO-TICKET] Created ticket {ticket_id} for low confidence response (confidence: {confidence:.3f}, domain: {query_domain}, status: {status})")
                if agent_assigned:
                    print(f"[AUTO-TICKET] Auto-assigned to agent {agent_assigned}")
                
                # Broadcast ticket creation to agents via WebSocket
                try:
                    await manager.broadcast_ticket_update(ticket_id, "created", {
                        "ticket_id": ticket_id,
                        "status": status,
                        "agent_id": str(agent_assigned) if agent_assigned else None,
                        "domain": query_domain
                    })
                except Exception as e:
                    print(f"[WS] Error broadcasting ticket creation: {e}")
                
                # Replace answer with routing message when confidence is low
                # Translate to Urdu if the user's query was in Urdu
                routing_message_en = "I don't have enough information to answer this. I am routing you to a human agent."
                if is_urdu:
                    try:
                        # Import translation function from RAG module
                        from RAG_supabase import translate_english_to_urdu
                        final_answer = translate_english_to_urdu(routing_message_en)
                    except Exception as e:
                        print(f"[WARNING] Failed to translate routing message to Urdu: {e}")
                        final_answer = routing_message_en
                else:
                    final_answer = routing_message_en
        
        # Prepare response and convert all numpy types to native Python types
        response_data = {
            "answer": final_answer,
            "sources": sources if not should_create_ticket else [],  # Don't show sources when routing to agent
            "agent_chat": ticket_created,  # True if ticket was created
            "response_id": str(response["response_id"]),
            "confidence": confidence,
            "ticket_id": ticket_id if ticket_created else None
        }
        
        # Convert all numpy types in the response for JSON serialization
        return convert_numpy_types(response_data)
    except FileNotFoundError as e:
        print(f"[ERROR] FileNotFoundError in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Knowledge base not found in Supabase. Set ALKHIDMAT_ZIP_PATH and restart to build it.",
        )
    except Exception as exc:
        print(f"[ERROR] Exception in chat endpoint: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing chat message: {str(exc)}"
        )


# ============================================================================
# TICKET MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/tickets")
async def create_ticket_endpoint(
    req: TicketRequest,
    session_token: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Create a new ticket for human agent assistance."""
    # Prefer X-Session-ID header if provided, otherwise use request body
    session_to_resolve = session_token or req.session_id
    
    if not session_to_resolve:
        raise HTTPException(status_code=401, detail="Session ID required. Please provide X-Session-ID header or session_id in request body.")
    
    db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
    if not db_session_id:
        raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
    # Get session
    db_session = get_session(db_session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get domain from session or query if available
    # Try to get domain from recent queries/responses in this session
    from supabase_client import get_supabase_client
    supabase = get_supabase_client()
    recent_response = supabase.table("responses").select("domain").eq(
        "session_id", db_session_id
    ).order("timestamp", desc=True).limit(1).execute()
    
    domain = None
    if recent_response.data and recent_response.data[0].get("domain"):
        domain = recent_response.data[0].get("domain")
    
    # Create a response for the ticket
    response = create_response(
        db_session_id,
        req.initial_message or "User requested human agent assistance",
        domain=domain
    )
    
    # Create ticket with domain routing
    ticket = create_ticket(response["response_id"], domain=domain)
    
    return {
        "ticket_id": str(ticket.get("ticket_id")),
        "status": ticket.get("status"),
        "response_id": str(response.get("response_id"))
    }


@app.get("/tickets")
async def list_tickets_endpoint(
    status: Optional[str] = Query(None, description="Filter by status: active, in_progress, resolved"),
    unassigned: bool = Query(False, description="Show only unassigned tickets"),
    agent: dict = Depends(get_current_agent)
):
    """List all tickets (for agent dashboard)."""
    # If unassigned=True, show unassigned tickets, otherwise show agent's tickets
    if unassigned:
        tickets = list_tickets(unassigned_only=True)
    else:
        tickets = list_tickets(status=status, agent_id=str(agent["agent_id"]))
    
    # Separate assigned vs unassigned
    assigned = [t for t in tickets if t.get("is_assigned", False)]
    unassigned_list = [t for t in tickets if not t.get("is_assigned", False)]
    
    return {
        "tickets": tickets,
        "assigned": assigned,
        "unassigned": unassigned_list,
        "total": len(tickets)
    }


@app.get("/tickets/{ticket_id}")
async def get_ticket_endpoint(ticket_id: str, agent: dict = Depends(get_current_agent)):
    """Get ticket details."""
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get session and queries/responses
    response = ticket.get("response")
    session = None
    if response:
        session_id = response.get("session_id") if isinstance(response, dict) else response[0].get("session_id") if isinstance(response, list) else None
        if session_id:
            session = get_session(session_id)
    
    response_data = None
    if response:
        if isinstance(response, dict):
            response_data = {
                "response_id": response.get("response_id"),
                "content": response.get("content"),
                "timestamp": response.get("timestamp")
            }
        elif isinstance(response, list) and response:
            response_data = {
                "response_id": response[0].get("response_id"),
                "content": response[0].get("content"),
                "timestamp": response[0].get("timestamp")
            }
    
    return {
        "ticket_id": ticket.get("ticket_id"),
        "status": ticket.get("status"),
        "agent_id": ticket.get("agent_id"),
        "created_at": ticket.get("created_at"),
        "resolved_at": ticket.get("resolved_at"),
        "response": response_data,
        "session": {
            "session_id": session.get("session_id"),
            "user_id": session.get("user_id"),
            "started_at": session.get("started_at")
        } if session else None
    }


@app.post("/tickets/{ticket_id}/assign")
async def assign_ticket_endpoint(
    ticket_id: str,
    agent: dict = Depends(get_current_agent)
):
    """Assign a ticket to the current agent."""
    if not assign_ticket(ticket_id, agent["agent_id"]):
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Broadcast ticket assignment to all agents
    try:
        await manager.broadcast_ticket_update(ticket_id, "assigned", {
            "ticket_id": ticket_id,
            "status": "in_progress",
            "agent_id": str(agent["agent_id"])
        })
    except Exception as e:
        print(f"[WS] Error broadcasting ticket assignment: {e}")
    
    return {"status": "assigned", "ticket_id": ticket_id, "agent_id": str(agent["agent_id"])}


@app.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket_endpoint(
    ticket_id: str,
    agent: dict = Depends(get_current_agent)
):
    """Mark a ticket as resolved."""
    if not resolve_ticket(ticket_id):
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Broadcast ticket resolution to all agents
    try:
        await manager.broadcast_ticket_update(ticket_id, "resolved", {
            "ticket_id": ticket_id,
            "status": "resolved"
        })
    except Exception as e:
        print(f"[WS] Error broadcasting ticket resolution: {e}")
    
    return {"status": "resolved", "ticket_id": ticket_id}


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.get("/admin/analytics")
async def get_analytics(admin: dict = Depends(get_current_admin)):
    """Get analytics for admin dashboard."""
    analytics = get_ticket_analytics()
    return analytics


@app.get("/admin/tickets")
async def admin_list_tickets(
    status: Optional[str] = Query(None),
    admin: dict = Depends(get_current_admin)
):
    """List all tickets (for admin)."""
    tickets = list_tickets(status=status)
    return {"tickets": tickets}


@app.post("/admin/kb/update")
async def update_knowledge_base(
    reindex_existing: bool = Query(False, description="Re-index existing documents"),
    admin: dict = Depends(get_current_admin)
):
    """
    Manually trigger incremental knowledge base update.
    Processes new documents from the ZIP file.
    """
    if not Path(ZIP_PATH).exists():
        raise HTTPException(
            status_code=404,
            detail=f"ZIP file not found at {ZIP_PATH}. Set ALKHIDMAT_ZIP_PATH environment variable."
        )
    
    try:
        stats = await asyncio.to_thread(
            rag_module.add_documents_from_zip_incremental,
            ZIP_PATH,
            reindex_existing=reindex_existing
        )
        
        return {
            "status": "success",
            "message": "Knowledge base update completed",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating knowledge base: {str(e)}"
        )


# ============================================================================
# LEGACY ENDPOINTS (for backward compatibility)
# ============================================================================

@app.post("/tickets/{ticket_id}/message")
async def send_agent_message(ticket_id: str, req: MessageRequest, agent: dict = Depends(get_current_agent)):
    """Send a message in agent chat. Stores message in database."""
    from supabase_client import get_supabase_client
    
    # Get ticket to find the session
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get session ID from ticket response
    response = ticket.get("response")
    if not response:
        raise HTTPException(status_code=404, detail="Response not found for this ticket")
    
    # Handle both dict and list response formats
    if isinstance(response, dict):
        session_id = response.get("session_id")
    elif isinstance(response, list) and len(response) > 0:
        session_id = response[0].get("session_id")
    else:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Store agent message as a response in the database
    # Agent messages are stored as responses with sender='agent'
    agent_response = create_response(
        session_id,
        req.message,
        confidence=1.0,  # Agent messages have high confidence
        domain=None  # Domain already set on ticket
    )
    
    print(f"[AGENT-MESSAGE] Agent {agent.get('agent_id')} sent message to ticket {ticket_id}, stored as response {agent_response.get('response_id')}")
    
    # Broadcast message to user via WebSocket
    try:
        await manager.send_to_user(str(session_id), {
            "type": "new_message",
            "message": {
                "role": "agent",
                "sender": "agent",
                "content": req.message,
                "timestamp": datetime.now().isoformat(),
                "response_id": str(agent_response.get("response_id"))
            }
        })
        print(f"[WS] Broadcasted agent message to user session {session_id}")
    except Exception as e:
        print(f"[WS] Error broadcasting agent message: {e}")
    
    return {
        "status": "sent",
        "ticket_id": ticket_id,
        "response_id": str(agent_response.get("response_id")),
        "message": req.message
    }


@app.get("/tickets/{ticket_id}/chat")
async def get_agent_chat_endpoint(ticket_id: str, agent: dict = Depends(get_current_agent)):
    """Get chat messages for a ticket (legacy endpoint)."""
    from supabase_client import get_supabase_client
    
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get session ID from ticket response
    response = ticket.get("response")
    if not response:
        raise HTTPException(status_code=404, detail="Response not found for this ticket")
    
    session_id = response.get("session_id") if isinstance(response, dict) else response[0].get("session_id") if isinstance(response, list) else None
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get ticket's initial response_id to identify RAG vs agent messages
    ticket_response_id = None
    if isinstance(ticket.get("response"), dict):
        ticket_response_id = ticket.get("response").get("response_id")
    elif isinstance(ticket.get("response"), list) and len(ticket.get("response")) > 0:
        ticket_response_id = ticket.get("response")[0].get("response_id")
    
    # Get queries and responses for the session
    supabase = get_supabase_client()
    queries_result = supabase.table("queries").select("*").eq("session_id", session_id).order("timestamp").execute()
    responses_result = supabase.table("responses").select("*").eq("session_id", session_id).order("timestamp").execute()
    
    queries = queries_result.data or []
    responses = responses_result.data or []
    
    # Combine queries and responses chronologically
    all_items = []
    for q in queries:
        all_items.append({
            "type": "query",
            "role": "user",
            "sender": "user",
            "content": q.get("content"),
            "timestamp": q.get("timestamp"),
            "query_id": str(q.get("query_id"))
        })
    
    for r in responses:
        response_id = str(r.get("response_id"))
        # Identify if this is the ticket's initial RAG response or an agent message
        # Agent messages are responses created after ticket creation with confidence=1.0
        # and are NOT the ticket's initial response
        is_agent_message = (
            ticket_response_id and 
            response_id != str(ticket_response_id) and 
            r.get("confidence") == 1.0
        )
        
        sender = "agent" if is_agent_message else "assistant"
        
        all_items.append({
            "type": "response",
            "role": "assistant" if sender == "assistant" else "agent",
            "sender": sender,
            "content": r.get("content"),
            "timestamp": r.get("timestamp"),
            "response_id": response_id,
            "confidence": r.get("confidence"),
            "domain": r.get("domain")
        })
    
    all_items.sort(key=lambda x: x.get("timestamp", ""))
    
    return {
        "ticket_id": ticket_id,
        "messages": all_items
    }


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/user/{session_id}")
async def websocket_user_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for user chat - receives real-time agent messages"""
    # Verify session
    db_session_id, error_msg = resolve_session_id(session_id)
    if not db_session_id:
        await websocket.close(code=1008, reason="Invalid session")
        return
    
    await manager.connect_user(websocket, str(db_session_id))
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect_user(websocket, str(db_session_id))
    except Exception as e:
        print(f"[WS] Error in user websocket: {e}")
        manager.disconnect_user(websocket, str(db_session_id))


@app.websocket("/ws/agent/{agent_token}")
async def websocket_agent_endpoint(websocket: WebSocket, agent_token: str):
    """WebSocket endpoint for agent dashboard - receives real-time ticket updates"""
    # Verify agent token
    if agent_token not in agent_sessions:
        await websocket.close(code=1008, reason="Invalid agent token")
        return
    
    agent_id = agent_sessions[agent_token]
    await manager.connect_agent(websocket, str(agent_id))
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect_agent(websocket, str(agent_id))
    except Exception as e:
        print(f"[WS] Error in agent websocket: {e}")
        manager.disconnect_agent(websocket, str(agent_id))
