# """
# FastAPI wrapper to serve the Alkhidmat RAG pipeline using RAG_supabase.py.
# - Authentication endpoints for users (OTP), agents, and admins
# - Chat endpoints with session management
# - Ticket management endpoints
# - Analytics endpoints for admins

# Environment variables:
# - SUPABASE_URL         : Supabase project URL
# - SUPABASE_KEY         : Supabase anon key (or SUPABASE_ANON_KEY)
# - ALKHIDMAT_ZIP_PATH   : path to the knowledge base zip (default: "Al Khidmat Knowledge Base.zip")
#                           Only needed for initial KB build
# """

# import asyncio
# import os
# import json
# import time
# from pathlib import Path
# from typing import Optional, Tuple, Any, Dict, List
# from datetime import datetime, timedelta
# import numpy as np

# # img processing imports 
# from fastapi import File, Form, UploadFile
# from ocr_utils import extract_text_from_image

# from fastapi import FastAPI, HTTPException, Query, Depends, Header, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import json

# from supabase_client import init_supabase
# from db_operations import (
#     create_user, get_user_by_phone, get_user_by_id, generate_otp, verify_otp,
#     create_session, get_session, update_session_activity,
#     create_query, update_query_domain, create_response, create_ticket, get_ticket, list_tickets,
#     assign_ticket, resolve_ticket, authenticate_agent, authenticate_admin,
#     get_ticket_analytics, get_agent, get_admin,
#     get_user_chat_history, get_session_chat_history
# )

# # Import RAG_supabase module
# import RAG_supabase as rag_module

# ZIP_PATH = os.getenv("ALKHIDMAT_ZIP_PATH", "Al Khidmat Knowledge Base.zip")

# app = FastAPI(title="Alkhidmat RAG API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory session storage (in production, use Redis or JWT tokens)
# user_sessions: dict = {}  # {session_token: {"user_id": user_id, "db_session_id": db_session_id}}
# agent_sessions: dict = {}  # {token: agent_id}
# admin_sessions: dict = {}  # {token: admin_id}

# # Track pending agent confirmation requests
# pending_agent_confirmations: dict = {}  # {session_id: {"pending": True, "timestamp": float}}

# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# def convert_numpy_types(obj: Any) -> Any:
#     """Recursively convert numpy types to native Python types for JSON serialization."""
#     if isinstance(obj, (np.integer, np.floating)):
#         return obj.item()  # Convert numpy scalar to Python native type
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()  # Convert numpy array to list
#     elif isinstance(obj, dict):
#         return {key: convert_numpy_types(value) for key, value in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return [convert_numpy_types(item) for item in obj]
#     else:
#         return obj

# # ============================================================================
# # WEBSOCKET CONNECTION MANAGER
# # ============================================================================

# class ConnectionManager:
#     def __init__(self):
#         # {session_id: [websocket1, websocket2, ...]} for users
#         # {agent_id: [websocket1, websocket2, ...]} for agents
#         self.user_connections: dict = {}
#         self.agent_connections: dict = {}
    
#     async def connect_user(self, websocket: WebSocket, session_id: str):
#         await websocket.accept()
#         if session_id not in self.user_connections:
#             self.user_connections[session_id] = []
#         self.user_connections[session_id].append(websocket)
#         print(f"[WS] User connected: {session_id} (total: {len(self.user_connections[session_id])})")
    
#     def disconnect_user(self, websocket: WebSocket, session_id: str):
#         if session_id in self.user_connections:
#             if websocket in self.user_connections[session_id]:
#                 self.user_connections[session_id].remove(websocket)
#             if len(self.user_connections[session_id]) == 0:
#                 del self.user_connections[session_id]
#         print(f"[WS] User disconnected: {session_id}")
    
#     async def send_to_user(self, session_id: str, message: dict):
#         if session_id in self.user_connections:
#             disconnected = []
#             for websocket in self.user_connections[session_id]:
#                 try:
#                     await websocket.send_json(message)
#                 except Exception as e:
#                     print(f"[WS] Error sending to user {session_id}: {e}")
#                     disconnected.append(websocket)
            
#             # Remove disconnected websockets
#             for ws in disconnected:
#                 self.user_connections[session_id].remove(ws)
    
#     async def connect_agent(self, websocket: WebSocket, agent_id: str):
#         await websocket.accept()
#         if agent_id not in self.agent_connections:
#             self.agent_connections[agent_id] = []
#         self.agent_connections[agent_id].append(websocket)
#         print(f"[WS] Agent connected: {agent_id} (total: {len(self.agent_connections[agent_id])})")
    
#     def disconnect_agent(self, websocket: WebSocket, agent_id: str):
#         if agent_id in self.agent_connections:
#             if websocket in self.agent_connections[agent_id]:
#                 self.agent_connections[agent_id].remove(websocket)
#             if len(self.agent_connections[agent_id]) == 0:
#                 del self.agent_connections[agent_id]
#         print(f"[WS] Agent disconnected: {agent_id}")
    
#     async def send_to_agent(self, agent_id: str, message: dict):
#         if agent_id in self.agent_connections:
#             disconnected = []
#             for websocket in self.agent_connections[agent_id]:
#                 try:
#                     await websocket.send_json(message)
#                 except Exception as e:
#                     print(f"[WS] Error sending to agent {agent_id}: {e}")
#                     disconnected.append(websocket)
            
#             # Remove disconnected websockets
#             for ws in disconnected:
#                 self.agent_connections[agent_id].remove(ws)
    
#     async def broadcast_ticket_update(self, ticket_id: str, update_type: str, data: dict):
#         """Broadcast ticket updates to all connected agents"""
#         message = {
#             "type": "ticket_update",
#             "ticket_id": ticket_id,
#             "update_type": update_type,
#             "data": data
#         }
#         # Broadcast to all agents
#         for agent_id, connections in list(self.agent_connections.items()):
#             for websocket in connections:
#                 try:
#                     await websocket.send_json(message)
#                 except Exception as e:
#                     print(f"[WS] Error broadcasting to agent {agent_id}: {e}")

# manager = ConnectionManager()

# # ============================================================================
# # REQUEST MODELS
# # ============================================================================

# class ChatRequest(BaseModel):
#     message: str


# class OTPRequest(BaseModel):
#     phone_number: str


# class OTPVerifyRequest(BaseModel):
#     phone_number: str
#     otp: str


# class AgentLoginRequest(BaseModel):
#     email: str
#     password: str


# class AdminLoginRequest(BaseModel):
#     email: str
#     password: str


# class TicketRequest(BaseModel):
#     session_id: str
#     initial_message: str = ""


# class MessageRequest(BaseModel):
#     message: str
#     sender: str = "agent"  # "agent" or "user"


# # ============================================================================
# # AUTHENTICATION HELPERS
# # ============================================================================

# def get_current_user(session_id: str = Header(..., alias="X-Session-ID")):
#     """Get current user from session."""
#     if session_id not in user_sessions:
#         raise HTTPException(status_code=401, detail="Invalid session")
#     session_data = user_sessions[session_id]
#     user_id = session_data.get("user_id") if isinstance(session_data, dict) else session_data
#     user = get_user_by_id(user_id)
#     if not user:
#         raise HTTPException(status_code=401, detail="User not found")
#     return user


# def get_current_agent(token: str = Header(..., alias="X-Agent-Token")):
#     """Get current agent from token."""
#     if token not in agent_sessions:
#         raise HTTPException(status_code=401, detail="Invalid agent token")
#     agent_id = agent_sessions[token]
#     agent = get_agent(agent_id)
#     if not agent:
#         raise HTTPException(status_code=401, detail="Agent not found")
#     return agent


# def get_current_admin(token: str = Header(..., alias="X-Admin-Token")):
#     """Get current admin from token."""
#     if token not in admin_sessions:
#         raise HTTPException(status_code=401, detail="Invalid admin token")
#     admin_id = admin_sessions[token]
#     admin = get_admin(admin_id)
#     if not admin:
#         raise HTTPException(status_code=401, detail="Admin not found")
#     return admin


# # ============================================================================
# # STARTUP
# # ============================================================================

# def _should_check_for_updates() -> bool:
#     """
#     Check if it's time to check for KB updates (weekly basis).
#     Returns True if a week has passed since last check, False otherwise.
#     """
#     timestamp_file = Path(".kb_last_check.json")
#     check_interval_days = 7  # Weekly check
    
#     if not timestamp_file.exists():
#         # First time - create timestamp file and return True
#         timestamp_file.write_text(json.dumps({
#             "last_check": datetime.now().isoformat()
#         }))
#         return True
    
#     try:
#         data = json.loads(timestamp_file.read_text())
#         last_check_str = data.get("last_check")
#         if not last_check_str:
#             return True
        
#         last_check = datetime.fromisoformat(last_check_str)
#         days_since_check = (datetime.now() - last_check).days
        
#         if days_since_check >= check_interval_days:
#             # Update timestamp
#             timestamp_file.write_text(json.dumps({
#                 "last_check": datetime.now().isoformat()
#             }))
#             return True
#         else:
#             days_remaining = check_interval_days - days_since_check
#             print(f"  KB update check skipped (last checked {days_since_check} day(s) ago, next check in {days_remaining} day(s))")
#             return False
#     except Exception as e:
#         print(f"  ⚠️ Error reading KB check timestamp: {e}")
#         # On error, allow check to proceed
#         return True

# def _ensure_index_exists():
#     """Check if knowledge base exists in Supabase, build if missing, update incrementally if exists."""
#     from supabase_client import get_supabase_client
    
#     # Check if ZIP file exists
#     if not Path(ZIP_PATH).exists():
#         print(f"  ZIP file not found at {ZIP_PATH}")
#         print("   Please set ALKHIDMAT_ZIP_PATH environment variable or place the ZIP file in the expected location.")
#         print("   The server will start, but RAG queries will fail until the knowledge base is built.")
#         return
    
#     try:
#         # Check if documents exist in Supabase
#         supabase = get_supabase_client()
#         if supabase:
#             result = supabase.table("documents").select("doc_id", count="exact").limit(1).execute()
#             doc_count = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
            
#             if doc_count > 0:
#                 print(f"✓ Knowledge base found in Supabase ({doc_count} documents).")
                
#                 # Only check for updates on a weekly basis
#                 if _should_check_for_updates():
#                     print(f"  Checking for new documents in ZIP file (weekly check)...")
#                     # Run incremental update to add any new documents
#                     try:
#                         stats = rag_module.add_documents_from_zip_incremental(ZIP_PATH, reindex_existing=False)
#                         if stats["added"] > 0:
#                             print(f"✓ Added {stats['added']} new document(s) to knowledge base")
#                         if stats["skipped"] > 0:
#                             print(f"  Skipped {stats['skipped']} existing document(s)")
#                         if stats["failed"] > 0:
#                             print(f"⚠️ Failed to add {stats['failed']} document(s)")
#                     except Exception as e:
#                         print(f"⚠️ Error during incremental update: {e}")
#                         print("   Knowledge base exists, but new documents may not have been added.")
#                 else:
#                     print(f"  KB update check skipped (weekly schedule)")
#                 return
#         else:
#             print("  Supabase client not initialized. Skipping knowledge base check.")
#             return
#     except Exception as e:
#         print(f"  Could not check Supabase documents: {e}")
#         print("   The server will start, but RAG queries may fail if the knowledge base is not built.")
#         return
    
#     # No documents found, need to build from scratch
#     print(f"  Knowledge base not found in Supabase. Building now...")
#     try:
#         # Build from scratch (not incremental since there's nothing to increment)
#         rag_module.build_alkhidmat_rag(ZIP_PATH, clear_existing=False, incremental=False)
#         print(f"✓ Knowledge base built successfully in Supabase")
#         # Update timestamp after initial build
#         timestamp_file = Path(".kb_last_check.json")
#         timestamp_file.write_text(json.dumps({
#             "last_check": datetime.now().isoformat()
#         }))
#     except Exception as e:
#         print(f"  Error building knowledge base: {e}")
#         print("   The server will start, but RAG queries will fail until the knowledge base is built.")


# @app.on_event("startup")
# async def _startup():
#     # Initialize Supabase client (non-blocking - will warn if not configured)
#     init_supabase()
#     # Build index in a thread to avoid blocking the event loop on startup.
#     await asyncio.to_thread(_ensure_index_exists)
    
#     # Pre-initialize domain embeddings to avoid delay on first query
#     print("[STARTUP] Pre-initializing domain embeddings...", flush=True)
#     try:
#         from RAG_supabase import DomainClassifier
#         DomainClassifier.initialize_domain_embeddings()
#         print("[STARTUP] ✓ Domain embeddings ready", flush=True)
#     except Exception as e:
#         print(f"[STARTUP]   Could not pre-initialize domain embeddings: {e}", flush=True)
    
#     # Pre-load embedding model to avoid delay on first query
#     print("[STARTUP] Pre-loading embedding model...", flush=True)
#     try:
#         from RAG_supabase import get_embedder
#         get_embedder()
#         print("[STARTUP] ✓ Embedding model ready", flush=True)
#     except Exception as e:
#         print(f"[STARTUP]   Could not pre-load embedding model: {e}", flush=True)


# @app.get("/health")
# def health():
#     return {"status": "ok"}


# # ============================================================================
# # USER AUTHENTICATION ENDPOINTS
# # ============================================================================

# @app.post("/auth/user/send-otp")
# async def send_otp(req: OTPRequest):
#     """Send OTP to phone number."""
#     otp = generate_otp(req.phone_number)
#     # In production, send OTP via SMS service
#     print(f"OTP for {req.phone_number}: {otp}")  # For development only
#     return {"message": "OTP sent", "otp": otp}  # Remove otp in production


# @app.post("/auth/user/verify-otp")
# async def verify_otp_endpoint(req: OTPVerifyRequest):
#     """Verify OTP and create/login user."""
#     if not verify_otp(req.phone_number, req.otp):
#         raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
#     # Get or create user
#     user = get_user_by_phone(req.phone_number)
#     if not user:
#         user = create_user(req.phone_number)
    
#     # Create new session for this login
#     session = create_session(user["id"])
#     db_session_id = str(session["session_id"])
    
#     # Use database session_id as the session token (it's a UUID)
#     # This way it persists across server restarts
#     session_token = db_session_id
    
#     # Store in memory for quick lookup
#     user_sessions[session_token] = {
#         "user_id": user["id"],
#         "db_session_id": session["session_id"]
#     }
    
#     # Get all chat history for this user (across all sessions) - WhatsApp-like behavior
#     chat_history = get_user_chat_history(user["id"])
    
#     return {
#         "session_id": session_token,  # This is now the database session_id (UUID)
#         "user_id": str(user["id"]),
#         "session": {
#             "session_id": str(session["session_id"]),
#             "started_at": session.get("started_at") or None
#         },
#         "chat_history": chat_history  # Include all previous messages
#     }


# # ============================================================================
# # AGENT AUTHENTICATION ENDPOINTS
# # ============================================================================

# @app.post("/auth/agent/login")
# async def agent_login(req: AgentLoginRequest):
#     """Login agent."""
#     agent = authenticate_agent(req.email, req.password)
#     if not agent:
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     import secrets
#     token = secrets.token_urlsafe(32)
#     agent_sessions[token] = agent["agent_id"]
    
#     return {
#         "token": token,
#         "agent_id": str(agent["agent_id"]),
#         "name": f"{agent['first_name']} {agent['last_name']}",
#         "email": agent["email"],
#         "domain": agent.get("domain")
#     }


# # ============================================================================
# # ADMIN AUTHENTICATION ENDPOINTS
# # ============================================================================

# @app.post("/auth/admin/login")
# async def admin_login(req: AdminLoginRequest):
#     """Login admin."""
#     admin = authenticate_admin(req.email, req.password)
#     if not admin:
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     import secrets
#     token = secrets.token_urlsafe(32)
#     admin_sessions[token] = admin["admin_id"]
    
#     return {
#         "token": token,
#         "admin_id": str(admin["admin_id"]),
#         "name": f"{admin['first_name']} {admin['last_name']}",
#         "email": admin["email"]
#     }


# # ============================================================================
# # SESSION RESOLUTION HELPER
# # ============================================================================

# def resolve_session_id(session_id: str) -> Tuple[Optional[str], Optional[str]]:
#     """
#     Resolve a session_id (which could be an old token or a UUID) to a database session_id.
#     Returns (db_session_id, error_message) tuple.
#     - If successful: (db_session_id, None)
#     - If failed: (None, error_message)
#     """
#     if not session_id:
#         return None, "Session ID is required"
    
#     # Check in-memory cache first
#     if session_id in user_sessions:
#         session_data = user_sessions[session_id]
#         if isinstance(session_data, dict):
#             return session_data.get("db_session_id"), None
#         return session_id, None
    
#     # Check if it's a valid UUID format
#     from uuid import UUID
#     is_valid_uuid = False
#     try:
#         UUID(session_id)
#         is_valid_uuid = True
#     except (ValueError, TypeError):
#         pass
    
#     if is_valid_uuid:
#         # Try to look up in database
#         try:
#             db_session = get_session(session_id)
#             if db_session:
#                 # Restore to in-memory cache
#                 user_sessions[session_id] = {
#                     "user_id": db_session["user_id"],
#                     "db_session_id": db_session["session_id"]
#                 }
#                 return db_session["session_id"], None
#             else:
#                 return None, "Session not found in database. Please login again."
#         except Exception as e:
#             return None, f"Error looking up session: {str(e)}"
#     else:
#         # Old token format - not stored in database, only in memory (lost on restart)
#         return None, "Session expired (old token format). Please login again to get a new session."


# # ============================================================================
# # CHAT ENDPOINTS
# # ============================================================================

# @app.post("/chats")
# async def create_chat(session_token: Optional[str] = Header(None, alias="X-Session-ID")):
#     """Create a new chat session and return chat history."""
#     if not session_token:
#         raise HTTPException(status_code=401, detail="Please login first. Session token required in X-Session-ID header.")
    
#     db_session_id, error_msg = resolve_session_id(session_token)
    
#     if not db_session_id:
#         raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
#     # Get session details
#     db_session = get_session(db_session_id)
#     if not db_session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     user_id = db_session["user_id"]
    
#     # Get all chat history for this user (across all sessions)
#     chat_history = get_user_chat_history(user_id)
    
#     return {
#         "session_id": session_token,  # Return the token that was sent (could be UUID or old format)
#         "db_session_id": str(db_session_id),
#         "user_id": str(user_id),
#         "chat_history": chat_history  # Include all previous messages
#     }


# @app.get("/chats/{session_id}/history")
# async def get_chat_history(
#     session_id: str,
#     session_token: Optional[str] = Header(None, alias="X-Session-ID")
# ):
#     """Get chat history for a session."""
#     session_to_resolve = session_token or session_id
    
#     db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
#     if not db_session_id:
#         raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
#     # Get chat history for this specific session
#     chat_history = get_session_chat_history(db_session_id)
    
#     return {
#         "session_id": str(db_session_id),
#         "messages": chat_history
#     }


# # @app.post("/chats/{session_id}")
# # async def chat(
# #     session_id: str, 
# #     req: ChatRequest,
# #     session_token: Optional[str] = Header(None, alias="X-Session-ID")
# # ):

# # refined for image processing 
# @app.post("/chats/{session_id}")
# async def chat(
#     session_id: str,
#     message: str = Form(""),
#     image: UploadFile | None = File(None),
#     session_token: Optional[str] = Header(None, alias="X-Session-ID")
# ):

#     """Process chat message with RAG."""
#     # Prefer X-Session-ID header if provided, otherwise use URL parameter
#     session_to_resolve = session_token or session_id
    
#     # Debug logging
#     print(f"[DEBUG] Chat request - URL session_id: {session_id}, Header X-Session-ID: {session_token}, Resolving: {session_to_resolve}")
    
#     db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
#     if not db_session_id:
#         print(f"[DEBUG] Session resolution failed: {error_msg}")
#         raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
#     # Get or verify session
#     db_session = get_session(db_session_id)
#     if not db_session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     # Update session activity
#     update_session_activity(db_session_id)
    
#     # add image_processing step
#     final_message = message.strip()
#     print(f"[IMAGE-DEBUG] User text: '{message}'")
#     print(f"[IMAGE-DEBUG] Image provided: {image is not None}")
#     if image:
#         image_bytes = await image.read()
#         ocr_text = extract_text_from_image(image_bytes)
#         print(f"[IMAGE-DEBUG] OCR result length: {len(ocr_text)} chars")
#         if ocr_text:
#             final_message += (
#                 "\n\nText detected in attached image:\n"
#                 + ocr_text
#             )

    
#     # Create query record (domain will be updated after RAG processing)
#     query_record = create_query(db_session_id, final_message)
#     query_id = query_record["query_id"] if query_record else None
    
#     # Check if user message confirms or denies agent connection
#     def is_agent_confirmation(message: str) -> bool:
#         """Check if message is a confirmation to connect with agent."""
#         if not message:
#             return False
        
#         message_clean = message.lower().strip()
#         confirmation_keywords = [
#             "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "fine",
#             "go ahead", "connect", "please", "proceed", "do it", "confirm",
#             "haan", "ji haan", "bilkul", "theek hai", "accha", "ok"  # Urdu/Hindi
#         ]
#         return any(keyword in message_clean for keyword in confirmation_keywords)
    
#     def is_agent_denial(message: str) -> bool:
#         """Check if message is a denial to connect with agent."""
#         if not message:
#             return False
        
#         message_clean = message.lower().strip()
#         denial_keywords = [
#             "no", "nope", "nah", "cancel", "stop", "don't", "dont", "not",
#             "nahi", "na", "mat", "cancel karo"  # Urdu/Hindi
#         ]
#         return any(keyword in message_clean for keyword in denial_keywords)
    
#     # Check if user message is a nicety/polite response
#     def is_nicety(message: str) -> bool:
#         """Detect if message is a nicety like thank you, goodbye, etc."""
#         if not message:
#             return False
        
#         message_original = message.lower().strip()
#         message_clean = message_original
        
#         # Check if it's a question FIRST (before removing punctuation)
#         # Questions are NOT niceties
#         question_words = ["what", "who", "where", "when", "how", "why", "which", "whose", "whom",
#                          "kya", "kaun", "kahan", "kab", "kyun", "kis", "kisne"]  # Urdu question words
        
#         # Check for question mark
#         has_question_mark = "?" in message_original
        
#         # Check if starts with question word
#         starts_with_question = any(message_original.startswith(q + " ") for q in question_words)
        
#         # Check if contains question patterns (e.g., "are you", "is it", "do you", "can you")
#         question_patterns = ["are you", "is it", "do you", "can you", "will you", "would you", 
#                             "have you", "has it", "did you", "does it", "was it", "were you"]
#         has_question_pattern = any(pattern in message_original for pattern in question_patterns)
        
#         # If it's clearly a question, it's NOT a nicety
#         if has_question_mark or starts_with_question or has_question_pattern:
#             return False
        
#         # Remove trailing punctuation for pattern matching
#         while message_clean and message_clean[-1] in '.,!?…':
#             message_clean = message_clean[:-1].strip()
        
#         nicety_patterns = [
#             # Thank you variations (exact phrases)
#             "thank you", "thanks", "thank", "thx", "ty",
#             "shukriya", "shukria", "shukar", "meherbani",  # Urdu
#             # Appreciation (only if standalone or with "i")
#             "i appreciate", "appreciate it", "much appreciated",
#             "mashkur", "shukar guzar",  # Urdu
#             # Goodbye (exact phrases)
#             "goodbye", "bye", "bye bye", "see you", "farewell",
#             "khuda hafiz", "allah hafiz", "fi amanillah",  # Urdu
#             # Positive feedback (only if standalone or with "that's", "it's", etc.)
#             "that's great", "that's good", "good job", "well done", "that's nice", "that's helpful",
#             "accha", "theek", "bohot accha",  # Urdu
#             # Welcome/acknowledgment (only standalone)
#             "ok", "okay", "alright", "got it", "understood",
#             "samajh gaya", "theek hai",  # Urdu
#         ]
        
#         # Check if message matches nicety patterns exactly or starts with them
#         is_nicety = False
#         for pattern in nicety_patterns:
#             # Exact match
#             if message_clean == pattern:
#                 is_nicety = True
#                 break
#             # Starts with pattern
#             if message_clean.startswith(pattern + " "):
#                 is_nicety = True
#                 break
#             # Ends with pattern (for phrases like "that's great")
#             if message_clean.endswith(" " + pattern):
#                 is_nicety = True
#                 break
        
#         # Also check if it's a very short message (1-2 words) that's likely a nicety
#         words = message_clean.split()
#         if not is_nicety and len(words) <= 2:
#             short_niceties = ["thanks", "thank", "thx", "ty", "bye", "ok", "okay", "shukriya"]
#             # Only match if the entire message is a short nicety
#             is_nicety = message_clean in short_niceties or (len(words) == 1 and words[0] in short_niceties)
        
#         return is_nicety
    
#     def get_nicety_response(message: str) -> str:
#         """Get appropriate response for nicety."""
#         if not message:
#             return "You're welcome! Is there anything else I can help you with?"
        
#         message_clean = message.lower().strip()
#         # Remove trailing punctuation
#         while message_clean and message_clean[-1] in '.,!?…':
#             message_clean = message_clean[:-1].strip()
        
#         # Thank you responses (must start with or be exact match, not just contain the word)
#         thank_keywords = ["thank you", "thanks", "thank", "thx", "ty", "shukriya", "shukria", "shukar", 
#                          "i appreciate", "appreciate it", "much appreciated", "grateful"]
#         if any(message_clean == kw or message_clean.startswith(kw + " ") for kw in thank_keywords):
#             return "You're welcome! I'm glad I could help. Is there anything else you'd like to know about Alkhidmat Foundation?"
        
#         # Goodbye responses (must start with or be exact match)
#         goodbye_keywords = ["bye", "bye bye", "goodbye", "see you", "farewell", "khuda hafiz", "allah hafiz"]
#         if any(message_clean == kw or message_clean.startswith(kw + " ") for kw in goodbye_keywords):
#             return "Goodbye! Feel free to come back if you have any questions about Alkhidmat Foundation. Have a great day!"
        
#         # Positive feedback (must be exact phrases, not just containing words)
#         positive_phrases = ["that's great", "that's good", "good job", "well done", "that's nice", "that's helpful", 
#                            "accha", "theek", "bohot accha"]
#         if any(message_clean == phrase or message_clean.startswith(phrase + " ") for phrase in positive_phrases):
#             return "Thank you! I'm happy to help. Is there anything else you'd like to know?"
        
#         # Default nicety response
#         return "You're welcome! Is there anything else I can help you with?"
    
#     # Check if user message requests human agent
#     def is_human_agent_request(message: str) -> bool:
#         """Detect if user message requests to connect with human agent."""
#         if not message:
#             return False
        
#         # Clean message: remove trailing punctuation and normalize
#         message_clean = message.lower().strip()
#         # Remove trailing punctuation (., !, ?, ...)
#         while message_clean and message_clean[-1] in '.,!?…':
#             message_clean = message_clean[:-1].strip()
        
#         # Core keywords that indicate agent request
#         core_keywords = [
#             "connect me with human",
#             "connect me with agent",
#             "connect with human",
#             "connect with agent",
#             "connect me to agent",
#             "connect me to human",
#             "connect to human",
#             "connect to agent",
#         ]
        
#         # Additional keywords
#         keywords = core_keywords + [
#             "human agent",
#             "talk to human",
#             "talk to agent",
#             "speak with human",
#             "speak with agent",
#             "chat with human",
#             "chat with agent",
#             "human help",
#             "agent help",
#             "need human",
#             "need agent",
#             "want human",
#             "want agent",
#             "transfer to human",
#             "transfer to agent",
#             "i want to talk to",
#             "can i speak to",
#             "can i talk to",
#             "let me talk to",
#             "let me speak to",
#             "i need to talk to",
#             "i need to speak to",
#             "i want to connect with agent",
#             "i want to connect with human",
#             "i need to connect with agent",
#             "i need to connect with human",
#             "want to talk to agent",
#             "want to talk to human",
#             "want to speak to agent",
#             "want to speak to human",
#             # Urdu keywords
#             "human se baat",
#             "agent se baat",
#             "human agent se",
#             "human ko connect",
#             "agent ko connect",
#         ]
        
#         # Check for matches
#         matched = any(keyword in message_clean for keyword in keywords)
        
#         # Also check if message starts with common patterns (more lenient)
#         if not matched:
#             starts_with_patterns = [
#                 "connect",
#                 "i want",
#                 "i need",
#                 "can i",
#                 "let me",
#             ]
#             contains_agent_keywords = ["agent", "human", "representative", "support"]
            
#             if any(message_clean.startswith(pattern) for pattern in starts_with_patterns):
#                 if any(keyword in message_clean for keyword in contains_agent_keywords):
#                     matched = True
        
#         if matched:
#             print(f"[HUMAN-AGENT-DETECTION] ✅ Detected human agent request in message: '{message}' (cleaned: '{message_clean}')")
#         else:
#             print(f"[HUMAN-AGENT-DETECTION] ❌ No match for message: '{message}' (cleaned: '{message_clean}')")
        
#         return matched
    
#     try:
#         # Check if user has an active ticket for this session
#         from supabase_client import get_supabase_client
#         supabase = get_supabase_client()
#         # Check for active tickets linked to responses in this session
#         active_responses = supabase.table("responses").select("response_id").eq(
#             "session_id", db_session_id
#         ).execute()
#         response_ids = [r["response_id"] for r in (active_responses.data or [])]
        
#         has_active_ticket = False
#         if response_ids:
#             active_tickets = supabase.table("tickets").select("ticket_id").in_(
#                 "response_id", response_ids
#             ).in_("status", ["active", "in_progress"]).execute()
#             has_active_ticket = len(active_tickets.data) > 0 if active_tickets.data else False
        
#         # Check if there's a pending agent confirmation request
#         pending_confirmation = pending_agent_confirmations.get(session_id)
#         if pending_confirmation:
#             # Check if user confirmed or denied (but not if it's just a nicety)
#             if is_agent_confirmation(req.message) and not is_nicety(req.message):
#                 print(f"[HUMAN-AGENT-CONFIRMATION] User confirmed: '{req.message}' - Routing to agent")
#                 # Clear pending confirmation
#                 del pending_agent_confirmations[session_id]
                
#                 # If there's already an active ticket, just inform user they're already connected
#                 if has_active_ticket:
#                     print(f"[HUMAN-AGENT-REQUEST] User already has active ticket - informing them")
#                     return {
#                         "answer": "You are already connected with a human agent. They will respond shortly.",
#                         "sources": [],
#                         "agent_chat": True,
#                         "response_id": None,
#                         "confidence": 1.0,
#                         "ticket_id": None
#                     }
                
#                 # Set domain to "general" for agent connection requests
#                 query_domain = "general"
#                 print(f"[HUMAN-AGENT-REQUEST] Setting domain to 'general' for agent connection request")
                
#                 # Update query with general domain
#                 if query_id:
#                     update_query_domain(query_id, query_domain)
#                     print(f"[QUERY-DOMAIN] Updated query {query_id} with domain: {query_domain}")
                
#                 # Create a placeholder response for the ticket
#                 placeholder_response = create_response(
#                     db_session_id,
#                     f"User confirmed to chat with human agent: {req.message}",
#                     confidence=1.0,
#                     domain=query_domain
#                 )
#                 # Create ticket with domain for better routing
#                 ticket = create_ticket(placeholder_response["response_id"], domain=query_domain)
#                 if ticket:
#                     ticket_id = str(ticket.get("ticket_id"))
#                     agent_assigned = ticket.get("agent_id")
#                     status = ticket.get("status")
#                     print(f"[HUMAN-AGENT-REQUEST] Created ticket {ticket_id} (status: {status}, domain: {query_domain})")
#                     if agent_assigned:
#                         print(f"[HUMAN-AGENT-REQUEST] Auto-assigned to agent {agent_assigned}")
                    
#                     # Return early with agent chat response - DO NOT PROCESS WITH RAG
#                     return {
#                         "answer": "I am connecting you with a human agent. They will respond shortly.",
#                         "sources": [],
#                         "agent_chat": True,
#                         "response_id": str(placeholder_response["response_id"]),
#                         "confidence": 1.0,
#                         "ticket_id": ticket_id
#                     }
#                 else:
#                     # If ticket creation failed, inform user and continue with RAG
#                     print(f"[HUMAN-AGENT-REQUEST] WARNING: Ticket creation failed")
#                     # Continue to RAG processing below
            
#             elif is_agent_denial(req.message):
#                 print(f"[HUMAN-AGENT-DENIAL] User declined: '{req.message}' - Continuing with RAG")
#                 # Clear pending confirmation and continue with normal RAG processing
#                 del pending_agent_confirmations[session_id]
#                 # Continue to RAG processing below
#             elif is_nicety(req.message):
#                 # User said a nicety while confirmation is pending - acknowledge but keep pending
#                 print(f"[HUMAN-AGENT-CONFIRMATION] User said nicety while confirmation pending: '{req.message}'")
#                 nicety_response = get_nicety_response(req.message)
#                 # Add reminder about pending confirmation
#                 reminder = " Also, would you like me to connect you with a human agent? Please reply with 'yes' to confirm or 'no' to cancel."
#                 return {
#                     "answer": nicety_response + reminder,
#                     "sources": [],
#                     "agent_chat": False,
#                     "response_id": None,
#                     "confidence": 1.0,
#                     "ticket_id": None
#                 }
#             else:
#                 # User didn't confirm or deny, might be asking something else
#                 # Clear pending confirmation if it's been more than 5 minutes
#                 if time.time() - pending_confirmation.get("timestamp", 0) > 300:
#                     del pending_agent_confirmations[session_id]
#                     print(f"[HUMAN-AGENT-CONFIRMATION] Pending confirmation expired, clearing")
#                 else:
#                     # Still waiting for confirmation, remind user
#                     return {
#                         "answer": "Would you like me to connect you with a human agent? Please reply with 'yes' to confirm or 'no' to cancel.",
#                         "sources": [],
#                         "agent_chat": False,
#                         "response_id": None,
#                         "confidence": 1.0,
#                         "ticket_id": None
#                     }
        
#         # Check if user explicitly requested human agent
#         user_requested_agent = is_human_agent_request(final_message)
#         print(f"[HUMAN-AGENT-CHECK] user_requested_agent={user_requested_agent}, has_active_ticket={has_active_ticket}")
        
#         # If user requested agent, ask for confirmation first
#         if user_requested_agent:
# <<<<<<< Updated upstream
#             print(f"[HUMAN-AGENT-REQUEST] User requested human agent: '{final_message}' - Routing immediately, skipping RAG")
# =======
#             print(f"[HUMAN-AGENT-REQUEST] User requested human agent: '{req.message}' - Asking for confirmation")
# >>>>>>> Stashed changes
            
#             # If there's already an active ticket, just inform user they're already connected
#             if has_active_ticket:
#                 print(f"[HUMAN-AGENT-REQUEST] User already has active ticket - informing them")
#                 return {
#                     "answer": "You are already connected with a human agent. They will respond shortly.",
#                     "sources": [],
#                     "agent_chat": True,
#                     "response_id": None,
#                     "confidence": 1.0,
#                     "ticket_id": None
#                 }
            
#             # Store pending confirmation request
#             pending_agent_confirmations[session_id] = {
#                 "pending": True,
#                 "timestamp": time.time()
#             }
            
#             # Ask for confirmation instead of routing immediately
#             return {
#                 "answer": "I can connect you with a human agent. Would you like me to do that? Please reply with 'yes' to confirm or 'no' to cancel.",
#                 "sources": [],
#                 "agent_chat": False,
#                 "response_id": None,
#                 "confidence": 1.0,
#                 "ticket_id": None
#             }
        
#         # Check if message is a nicety (thank you, goodbye, etc.)
#         if is_nicety(req.message):
#             print(f"[NICETY-DETECTION] Detected nicety: '{req.message}' - Responding appropriately")
#             nicety_response = get_nicety_response(req.message)
            
#             # Create a response record for the nicety
#             nicety_response_record = create_response(
#                 db_session_id,
# <<<<<<< Updated upstream
#                 f"User requested to chat with human agent: {final_message}",
#                 confidence=1.0,  # High confidence since it's an explicit request
#                 domain=query_domain
# =======
#                 nicety_response,
#                 confidence=1.0,
#                 domain="general"
# >>>>>>> Stashed changes
#             )
            
#             return {
#                 "answer": nicety_response,
#                 "sources": [],
#                 "agent_chat": False,
#                 "response_id": str(nicety_response_record["response_id"]) if nicety_response_record else None,
#                 "confidence": 1.0,
#                 "ticket_id": None
#             }
        
#         # Process with RAG (use Self-RAG if enabled)
#         use_selfrag = getattr(rag_module, 'SELFRAG_ENABLE', False)
#         if use_selfrag:
#             # generate_answer_selfrag returns (answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics)
#             # Pass session_id for conversation memory
#             result = await asyncio.to_thread(
#                 rag_module.generate_answer_selfrag, final_message, top_k=5, filter_category=None, session_id=str(db_session_id)
#             )
#             answer, _, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics = result
#             is_urdu = (input_lang == "ur" or input_lang == "roman_ur")
#         else:
#             # generate_answer returns (answer, query, is_urdu, sources, confidence_scores, domain_classification)
#             answer, _, is_urdu, sources, confidence_scores, domain_classification = await asyncio.to_thread(
#                 rag_module.generate_answer, final_message, top_k=5, filter_category=None
#             )
        
#         # Use confidence from RAG module (combined confidence score)
#         confidence = None
#         if confidence_scores and isinstance(confidence_scores, dict):
#             # Use combined confidence if available
#             confidence = confidence_scores.get("combined_confidence") or confidence_scores.get("retrieval_confidence")
#         elif isinstance(confidence_scores, (int, float)):
#             confidence = float(confidence_scores)
        
#         # Fallback: Calculate confidence from sources (average similarity score)
#         if confidence is None and sources and len(sources) > 0:
#             similarities = [s.get("similarity", 0.0) for s in sources if s.get("similarity") is not None]
#             if similarities:
#                 confidence = sum(similarities) / len(similarities)
        
#         # Convert confidence to native Python float if it's a numpy type
#         if confidence is not None:
#             confidence = convert_numpy_types(confidence)
        
#         # Use domain classification from RAG module
#         query_domain = None
#         if domain_classification and isinstance(domain_classification, dict):
#             # Get the primary domain (highest confidence)
#             if "primary_domain" in domain_classification:
#                 query_domain = domain_classification["primary_domain"]
#             elif "domain" in domain_classification:
#                 query_domain = domain_classification["domain"]
#         elif isinstance(domain_classification, str):
#             query_domain = domain_classification
        
#         # Fallback: Extract domain from first source (if available)
#         if not query_domain and sources and len(sources) > 0:
#             if sources[0].get("category"):
#                 query_domain = sources[0].get("category")
        
#         # Log domain classification for debugging
#         print(f"[DOMAIN-CLASSIFICATION] Domain classification result: {domain_classification}")
#         print(f"[DOMAIN-CLASSIFICATION] Extracted domain: '{query_domain}'")
        
#         # Update query record with domain if it was determined
#         if query_id and query_domain:
#             update_query_domain(query_id, query_domain)
#             print(f"[QUERY-DOMAIN] Updated query {query_id} with domain: {query_domain}")
        
#         # Create response record with confidence
#         response = create_response(
#             db_session_id, 
#             answer, 
#             confidence=confidence,
#             domain=query_domain
#         )
        
#         # Auto-create ticket if confidence is below threshold (0.5-0.6)
#         # Note: user_requested_agent is already handled above with early return
#         CONFIDENCE_THRESHOLD = 0.70  # Configurable threshold
#         ticket_created = False
#         ticket_id = None
#         final_answer = answer  # Default to original answer
        
#         # Create ticket if confidence is low (user-requested agent already handled above)
#         should_create_ticket = (
#             confidence is not None and confidence < CONFIDENCE_THRESHOLD
#         ) and not has_active_ticket
        
#         if should_create_ticket:
#             # Create ticket automatically with domain routing
#             ticket = create_ticket(response["response_id"], domain=query_domain)
#             if ticket:
#                 ticket_created = True
#                 ticket_id = str(ticket.get("ticket_id"))
#                 agent_assigned = ticket.get("agent_id")
#                 status = ticket.get("status")
#                 print(f"[AUTO-TICKET] Created ticket {ticket_id} for low confidence response (confidence: {confidence:.3f}, domain: {query_domain}, status: {status})")
#                 if agent_assigned:
#                     print(f"[AUTO-TICKET] Auto-assigned to agent {agent_assigned}")
                
#                 # Broadcast ticket creation to agents via WebSocket
#                 try:
#                     await manager.broadcast_ticket_update(ticket_id, "created", {
#                         "ticket_id": ticket_id,
#                         "status": status,
#                         "agent_id": str(agent_assigned) if agent_assigned else None,
#                         "domain": query_domain
#                     })
#                 except Exception as e:
#                     print(f"[WS] Error broadcasting ticket creation: {e}")
                
#                 # Replace answer with routing message when confidence is low
#                 # Translate to Urdu if the user's query was in Urdu
#                 routing_message_en = "I don't have enough information to answer this. I am routing you to a human agent."
#                 if is_urdu:
#                     try:
#                         # Import translation function from RAG module
#                         from RAG_supabase import translate_english_to_urdu
#                         final_answer = translate_english_to_urdu(routing_message_en)
#                     except Exception as e:
#                         print(f"[WARNING] Failed to translate routing message to Urdu: {e}")
#                         final_answer = routing_message_en
#                 else:
#                     final_answer = routing_message_en
        
#         # Prepare response and convert all numpy types to native Python types
#         response_data = {
#             "answer": final_answer,
#             "sources": sources if not should_create_ticket else [],  # Don't show sources when routing to agent
#             "agent_chat": ticket_created,  # True if ticket was created
#             "response_id": str(response["response_id"]),
#             "confidence": confidence,
#             "ticket_id": ticket_id if ticket_created else None
#         }
        
#         # Convert all numpy types in the response for JSON serialization
#         return convert_numpy_types(response_data)
#     except FileNotFoundError as e:
#         print(f"[ERROR] FileNotFoundError in chat endpoint: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500,
#             detail="Knowledge base not found in Supabase. Set ALKHIDMAT_ZIP_PATH and restart to build it.",
#         )
#     except Exception as exc:
#         print(f"[ERROR] Exception in chat endpoint: {type(exc).__name__}: {exc}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Error processing chat message: {str(exc)}"
#         )


# # ============================================================================
# # TICKET MANAGEMENT ENDPOINTS
# # ============================================================================

# @app.post("/tickets")
# async def create_ticket_endpoint(
#     req: TicketRequest,
#     session_token: Optional[str] = Header(None, alias="X-Session-ID")
# ):
#     """Create a new ticket for human agent assistance."""
#     # Prefer X-Session-ID header if provided, otherwise use request body
#     session_to_resolve = session_token or req.session_id
    
#     if not session_to_resolve:
#         raise HTTPException(status_code=401, detail="Session ID required. Please provide X-Session-ID header or session_id in request body.")
    
#     db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
#     if not db_session_id:
#         raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
#     # Get session
#     db_session = get_session(db_session_id)
#     if not db_session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     # Get domain from session or query if available
#     # Try to get domain from recent queries/responses in this session
#     from supabase_client import get_supabase_client
#     supabase = get_supabase_client()
#     recent_response = supabase.table("responses").select("domain").eq(
#         "session_id", db_session_id
#     ).order("timestamp", desc=True).limit(1).execute()
    
#     domain = None
#     if recent_response.data and recent_response.data[0].get("domain"):
#         domain = recent_response.data[0].get("domain")
    
#     # Create a response for the ticket
#     response = create_response(
#         db_session_id,
#         req.initial_message or "User requested human agent assistance",
#         domain=domain
#     )
    
#     # Create ticket with domain routing
#     ticket = create_ticket(response["response_id"], domain=domain)
    
#     return {
#         "ticket_id": str(ticket.get("ticket_id")),
#         "status": ticket.get("status"),
#         "response_id": str(response.get("response_id"))
#     }


# @app.get("/tickets")
# async def list_tickets_endpoint(
#     status: Optional[str] = Query(None, description="Filter by status: active, in_progress, resolved"),
#     unassigned: bool = Query(False, description="Show only unassigned tickets"),
#     agent: dict = Depends(get_current_agent)
# ):
#     """List all tickets (for agent dashboard)."""
#     # If unassigned=True, show unassigned tickets, otherwise show agent's tickets
#     if unassigned:
#         tickets = list_tickets(unassigned_only=True)
#     else:
#         tickets = list_tickets(status=status, agent_id=str(agent["agent_id"]))
    
#     # Separate assigned vs unassigned
#     assigned = [t for t in tickets if t.get("is_assigned", False)]
#     unassigned_list = [t for t in tickets if not t.get("is_assigned", False)]
    
#     return {
#         "tickets": tickets,
#         "assigned": assigned,
#         "unassigned": unassigned_list,
#         "total": len(tickets)
#     }


# @app.get("/tickets/{ticket_id}")
# async def get_ticket_endpoint(ticket_id: str, agent: dict = Depends(get_current_agent)):
#     """Get ticket details."""
#     ticket = get_ticket(ticket_id)
#     if not ticket:
#         raise HTTPException(status_code=404, detail="Ticket not found")
    
#     # Get session and queries/responses
#     response = ticket.get("response")
#     session = None
#     if response:
#         session_id = response.get("session_id") if isinstance(response, dict) else response[0].get("session_id") if isinstance(response, list) else None
#         if session_id:
#             session = get_session(session_id)
    
#     response_data = None
#     if response:
#         if isinstance(response, dict):
#             response_data = {
#                 "response_id": response.get("response_id"),
#                 "content": response.get("content"),
#                 "timestamp": response.get("timestamp")
#             }
#         elif isinstance(response, list) and response:
#             response_data = {
#                 "response_id": response[0].get("response_id"),
#                 "content": response[0].get("content"),
#                 "timestamp": response[0].get("timestamp")
#             }
    
#     return {
#         "ticket_id": ticket.get("ticket_id"),
#         "status": ticket.get("status"),
#         "agent_id": ticket.get("agent_id"),
#         "created_at": ticket.get("created_at"),
#         "resolved_at": ticket.get("resolved_at"),
#         "response": response_data,
#         "session": {
#             "session_id": session.get("session_id"),
#             "user_id": session.get("user_id"),
#             "started_at": session.get("started_at")
#         } if session else None
#     }


# @app.post("/tickets/{ticket_id}/assign")
# async def assign_ticket_endpoint(
#     ticket_id: str,
#     agent: dict = Depends(get_current_agent)
# ):
#     """Assign a ticket to the current agent."""
#     if not assign_ticket(ticket_id, agent["agent_id"]):
#         raise HTTPException(status_code=404, detail="Ticket not found")
    
#     # Broadcast ticket assignment to all agents
#     try:
#         await manager.broadcast_ticket_update(ticket_id, "assigned", {
#             "ticket_id": ticket_id,
#             "status": "in_progress",
#             "agent_id": str(agent["agent_id"])
#         })
#     except Exception as e:
#         print(f"[WS] Error broadcasting ticket assignment: {e}")
    
#     return {"status": "assigned", "ticket_id": ticket_id, "agent_id": str(agent["agent_id"])}


# @app.post("/tickets/{ticket_id}/resolve")
# async def resolve_ticket_endpoint(
#     ticket_id: str,
#     agent: dict = Depends(get_current_agent)
# ):
#     """Mark a ticket as resolved."""
#     if not resolve_ticket(ticket_id):
#         raise HTTPException(status_code=404, detail="Ticket not found")
    
#     # Broadcast ticket resolution to all agents
#     try:
#         await manager.broadcast_ticket_update(ticket_id, "resolved", {
#             "ticket_id": ticket_id,
#             "status": "resolved"
#         })
#     except Exception as e:
#         print(f"[WS] Error broadcasting ticket resolution: {e}")
    
#     return {"status": "resolved", "ticket_id": ticket_id}


# # ============================================================================
# # ADMIN ENDPOINTS
# # ============================================================================

# @app.get("/admin/analytics")
# async def get_analytics(admin: dict = Depends(get_current_admin)):
#     """Get analytics for admin dashboard."""
#     analytics = get_ticket_analytics()
#     return analytics


# @app.get("/admin/tickets")
# async def admin_list_tickets(
#     status: Optional[str] = Query(None),
#     admin: dict = Depends(get_current_admin)
# ):
#     """List all tickets (for admin)."""
#     tickets = list_tickets(status=status)
#     return {"tickets": tickets}


# @app.post("/admin/kb/update")
# async def update_knowledge_base(
#     reindex_existing: bool = Query(False, description="Re-index existing documents"),
#     admin: dict = Depends(get_current_admin)
# ):
#     """
#     Manually trigger incremental knowledge base update.
#     Processes new documents from the ZIP file.
#     """
#     if not Path(ZIP_PATH).exists():
#         raise HTTPException(
#             status_code=404,
#             detail=f"ZIP file not found at {ZIP_PATH}. Set ALKHIDMAT_ZIP_PATH environment variable."
#         )
    
#     try:
#         stats = await asyncio.to_thread(
#             rag_module.add_documents_from_zip_incremental,
#             ZIP_PATH,
#             reindex_existing=reindex_existing
#         )
        
#         return {
#             "status": "success",
#             "message": "Knowledge base update completed",
#             "stats": stats
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error updating knowledge base: {str(e)}"
#         )


# # ============================================================================
# # LEGACY ENDPOINTS (for backward compatibility)
# # ============================================================================

# @app.post("/tickets/{ticket_id}/message")
# async def send_agent_message(ticket_id: str, req: MessageRequest, agent: dict = Depends(get_current_agent)):
#     """Send a message in agent chat. Stores message in database."""
#     from supabase_client import get_supabase_client
    
#     # Get ticket to find the session
#     ticket = get_ticket(ticket_id)
#     if not ticket:
#         raise HTTPException(status_code=404, detail="Ticket not found")
    
#     # Get session ID from ticket response
#     response = ticket.get("response")
#     if not response:
#         raise HTTPException(status_code=404, detail="Response not found for this ticket")
    
#     # Handle both dict and list response formats
#     if isinstance(response, dict):
#         session_id = response.get("session_id")
#     elif isinstance(response, list) and len(response) > 0:
#         session_id = response[0].get("session_id")
#     else:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     if not session_id:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     # Store agent message as a response in the database
#     # Agent messages are stored as responses with sender='agent'
#     agent_response = create_response(
#         session_id,
#         req.message,
#         confidence=1.0,  # Agent messages have high confidence
#         domain=None  # Domain already set on ticket
#     )
    
#     print(f"[AGENT-MESSAGE] Agent {agent.get('agent_id')} sent message to ticket {ticket_id}, stored as response {agent_response.get('response_id')}")
    
#     # Broadcast message to user via WebSocket
#     try:
#         await manager.send_to_user(str(session_id), {
#             "type": "new_message",
#             "message": {
#                 "role": "agent",
#                 "sender": "agent",
#                 "content": req.message,
#                 "timestamp": datetime.now().isoformat(),
#                 "response_id": str(agent_response.get("response_id"))
#             }
#         })
#         print(f"[WS] Broadcasted agent message to user session {session_id}")
#     except Exception as e:
#         print(f"[WS] Error broadcasting agent message: {e}")
    
#     return {
#         "status": "sent",
#         "ticket_id": ticket_id,
#         "response_id": str(agent_response.get("response_id")),
#         "message": req.message
#     }


# @app.get("/tickets/{ticket_id}/chat")
# async def get_agent_chat_endpoint(ticket_id: str, agent: dict = Depends(get_current_agent)):
#     """Get chat messages for a ticket (legacy endpoint)."""
#     from supabase_client import get_supabase_client
    
#     ticket = get_ticket(ticket_id)
#     if not ticket:
#         raise HTTPException(status_code=404, detail="Ticket not found")
    
#     # Get session ID from ticket response
#     response = ticket.get("response")
#     if not response:
#         raise HTTPException(status_code=404, detail="Response not found for this ticket")
    
#     session_id = response.get("session_id") if isinstance(response, dict) else response[0].get("session_id") if isinstance(response, list) else None
#     if not session_id:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = get_session(session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     # Get ticket's initial response_id to identify RAG vs agent messages
#     ticket_response_id = None
#     if isinstance(ticket.get("response"), dict):
#         ticket_response_id = ticket.get("response").get("response_id")
#     elif isinstance(ticket.get("response"), list) and len(ticket.get("response")) > 0:
#         ticket_response_id = ticket.get("response")[0].get("response_id")
    
#     # Get queries and responses for the session
#     supabase = get_supabase_client()
#     queries_result = supabase.table("queries").select("*").eq("session_id", session_id).order("timestamp").execute()
#     responses_result = supabase.table("responses").select("*").eq("session_id", session_id).order("timestamp").execute()
    
#     queries = queries_result.data or []
#     responses = responses_result.data or []
    
#     # Combine queries and responses chronologically
#     all_items = []
#     for q in queries:
#         all_items.append({
#             "type": "query",
#             "role": "user",
#             "sender": "user",
#             "content": q.get("content"),
#             "timestamp": q.get("timestamp"),
#             "query_id": str(q.get("query_id"))
#         })
    
#     for r in responses:
#         response_id = str(r.get("response_id"))
#         # Identify if this is the ticket's initial RAG response or an agent message
#         # Agent messages are responses created after ticket creation with confidence=1.0
#         # and are NOT the ticket's initial response
#         is_agent_message = (
#             ticket_response_id and 
#             response_id != str(ticket_response_id) and 
#             r.get("confidence") == 1.0
#         )
        
#         sender = "agent" if is_agent_message else "assistant"
        
#         all_items.append({
#             "type": "response",
#             "role": "assistant" if sender == "assistant" else "agent",
#             "sender": sender,
#             "content": r.get("content"),
#             "timestamp": r.get("timestamp"),
#             "response_id": response_id,
#             "confidence": r.get("confidence"),
#             "domain": r.get("domain")
#         })
    
#     all_items.sort(key=lambda x: x.get("timestamp", ""))
    
#     return {
#         "ticket_id": ticket_id,
#         "messages": all_items
#     }


# # ============================================================================
# # WEBSOCKET ENDPOINTS
# # ============================================================================

# @app.websocket("/ws/user/{session_id}")
# async def websocket_user_endpoint(websocket: WebSocket, session_id: str):
#     """WebSocket endpoint for user chat - receives real-time agent messages"""
#     # Verify session
#     db_session_id, error_msg = resolve_session_id(session_id)
#     if not db_session_id:
#         await websocket.close(code=1008, reason="Invalid session")
#         return
    
#     await manager.connect_user(websocket, str(db_session_id))
    
#     try:
#         while True:
#             # Keep connection alive and handle any incoming messages
#             data = await websocket.receive_text()
#             try:
#                 message = json.loads(data)
#                 if message.get("type") == "ping":
#                     await websocket.send_json({"type": "pong"})
#             except json.JSONDecodeError:
#                 pass
#     except WebSocketDisconnect:
#         manager.disconnect_user(websocket, str(db_session_id))
#     except Exception as e:
#         print(f"[WS] Error in user websocket: {e}")
#         manager.disconnect_user(websocket, str(db_session_id))


# @app.websocket("/ws/agent/{agent_token}")
# async def websocket_agent_endpoint(websocket: WebSocket, agent_token: str):
#     """WebSocket endpoint for agent dashboard - receives real-time ticket updates"""
#     # Verify agent token
#     if agent_token not in agent_sessions:
#         await websocket.close(code=1008, reason="Invalid agent token")
#         return
    
#     agent_id = agent_sessions[agent_token]
#     await manager.connect_agent(websocket, str(agent_id))
    
#     try:
#         while True:
#             # Keep connection alive and handle any incoming messages
#             data = await websocket.receive_text()
#             try:
#                 message = json.loads(data)
#                 if message.get("type") == "ping":
#                     await websocket.send_json({"type": "pong"})
#             except json.JSONDecodeError:
#                 pass
#     except WebSocketDisconnect:
#         manager.disconnect_agent(websocket, str(agent_id))
#     except Exception as e:
#         print(f"[WS] Error in agent websocket: {e}")
#         manager.disconnect_agent(websocket, str(agent_id))
"""
main.py
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
from datetime import datetime, timedelta, timezone
import numpy as np

# img processing imports 
from fastapi import File, Form, UploadFile
from ocr_utils import extract_text_from_image

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

import rag_answer_cache

ZIP_PATH = os.getenv("ALKHIDMAT_ZIP_PATH", "Al Khidmat Knowledge Base.zip")

app = FastAPI(title="Alkhidmat RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

fb_svc_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
# Fallback to the specific file provided if env var is missing or invalid
if not fb_svc_path or not os.path.exists(fb_svc_path):
    potential_path = os.path.join(os.path.dirname(__file__), "fypp-ea6b9-firebase-adminsdk-fbsvc-1f21963d6d.json")
    if os.path.exists(potential_path):
        fb_svc_path = potential_path

if fb_svc_path and os.path.exists(fb_svc_path):
    try:
        cred = credentials.Certificate(fb_svc_path)
        firebase_admin.initialize_app(cred)
        print(f"[FIREBASE] Initialized with service account: {fb_svc_path}")
    except Exception as e:
        print(f"[FIREBASE] Initialization error: {e}")
else:
    print("[FIREBASE] SERVICE ACCOUNT NOT FOUND. Firebase login will be disabled.")

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


def _db_write_with_retry(fn, *args, max_attempts: int = 3, delay: float = 1.0, **kwargs):
    """
    Retry a Supabase write on transient network errors (ConnectError, getaddrinfo).
    Returns the result on success, or None after all retries are exhausted.
    A None return means the DB write failed non-fatally — the answer still reaches the user.
    """
    import time
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            is_network = any(kw in err for kw in ("getaddrinfo", "ConnectError", "ConnectionError", "timeout", "Timeout"))
            print(f"[DB-RETRY] Attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
            if attempt < max_attempts and is_network:
                time.sleep(delay * attempt)
            else:
                print(f"[DB-RETRY] Giving up after {attempt} attempt(s) — answer will still be returned to user.")
                return None

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
        print(f"[DEBUG-WS] Attempting to send to user: {session_id}")
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
            print(f"[DEBUG-WS] Successfully sent to user: {session_id}")
        else:
            print(f"[DEBUG-WS] User {session_id} not connected")
    
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


class FirebaseLoginRequest(BaseModel):
    id_token: str


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

    # ── Auto-resolve inactive tickets background task ────────────────────────
    asyncio.create_task(_auto_resolve_inactive_tickets())
    print("[STARTUP] ✓ Auto-resolve background task started", flush=True)


async def _auto_resolve_inactive_tickets():
    """
    Background task: every 5 minutes, resolve tickets where the user
    and agent have not replied for >= 15 minutes (configurable via AUTO_RESOLVE_MINUTES env).
    """
    INACTIVITY_MINUTES = int(os.getenv("AUTO_RESOLVE_MINUTES", "15"))
    CHECK_INTERVAL_SECONDS = 300  # 5 minutes

    print(f"[AUTO-RESOLVE] Background task running (threshold={INACTIVITY_MINUTES}min, check every {CHECK_INTERVAL_SECONDS}s)")
    while True:
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
        try:
            from supabase_client import get_supabase_client
            supabase = get_supabase_client()
            if not supabase:
                continue

            # Robust timezone-aware threshold
            now_utc = datetime.now(timezone.utc)
            inactivity_delta = timedelta(minutes=INACTIVITY_MINUTES)

            # Find in_progress / active tickets with no recent activity
            # Optimized: Fetch session_id via responses join in one query
            open_tickets_result = await asyncio.to_thread(
                lambda: supabase.table("tickets")
                .select("ticket_id, status, responses(session_id)")
                .in_("status", ["active", "in_progress"])
                .execute()
            )

            open_tickets = open_tickets_result.data or []
            if open_tickets:
                print(f"[AUTO-RESOLVE] Checking {len(open_tickets)} open tickets...")

            for t in open_tickets:
                ticket_id = t["ticket_id"]
                try:
                    # Extract session_id from the joined response
                    responses_data = t.get("responses")
                    session_id = None
                    if isinstance(responses_data, list) and responses_data:
                        session_id = responses_data[0].get("session_id")
                    elif isinstance(responses_data, dict):
                        session_id = responses_data.get("session_id")
                    
                    if not session_id:
                        continue

                    # Check last activity in this session (User Query or Agent Response)
                    # 1. Last User Query
                    q_res = await asyncio.to_thread(
                        lambda: supabase.table("queries")
                        .select("timestamp")
                        .eq("session_id", str(session_id))
                        .order("timestamp", desc=True)
                        .limit(1)
                        .execute()
                    )
                    
                    # 2. Last Agent Response
                    r_res = await asyncio.to_thread(
                        lambda: supabase.table("responses")
                        .select("timestamp")
                        .eq("session_id", str(session_id))
                        .order("timestamp", desc=True)
                        .limit(1)
                        .execute()
                    )

                    last_activity_ts = None
                    times = []
                    if q_res.data:
                        times.append(q_res.data[0].get("timestamp"))
                    if r_res.data:
                        times.append(r_res.data[0].get("timestamp"))
                    
                    if not times:
                        # No activity found (should not happen for a valid ticket), so we skip it
                        continue
                        
                    # Parse latest timestamp string to aware datetime
                    latest_str = max(times)
                    # Handle ISO format (supports 'Z' or '+00:00')
                    latest_dt = datetime.fromisoformat(latest_str.replace('Z', '+00:00'))
                    
                    # Compare
                    if (now_utc - latest_dt) > inactivity_delta:
                        print(f"[AUTO-RESOLVE] Resolving ticket {ticket_id} (inactive since {latest_dt})")
                        await asyncio.to_thread(resolve_ticket, str(ticket_id))

                        # Broadcast to agents
                        await manager.broadcast_ticket_update(str(ticket_id), "resolved", {
                            "ticket_id": str(ticket_id),
                            "status": "resolved",
                            "reason": "auto_resolved_inactivity"
                        })

                        # Notify user to return to AI
                        await manager.send_to_user(str(session_id), {
                            "type": "ticket_resolved",
                            "ticket_id": str(ticket_id),
                            "message": "Your support ticket has been closed due to inactivity. You are back with the AI assistant!"
                        })
                except Exception as te:
                    print(f"[AUTO-RESOLVE] Error processing ticket {ticket_id}: {te}")
        except Exception as e:
            print(f"[AUTO-RESOLVE] Loop error: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================================
# USER AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/user/send-otp")
async def send_otp(req: OTPRequest):
    """Send OTP to phone number via Twilio (SMS/WhatsApp) or terminal fallback."""
    otp = generate_otp(req.phone_number)

    # ── Twilio integration ───────────────────────────────────────────────────
    TWILIO_SID   = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_FROM  = os.getenv("TWILIO_PHONE_FROM")   # e.g. '+1XXXXXXXXXX' or 'whatsapp:+14155238886'
    TWILIO_USE_WHATSAPP = os.getenv("TWILIO_USE_WHATSAPP", "false").lower() in ("1", "true", "yes")

    if TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
        try:
            from twilio.rest import Client as TwilioClient
            client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
            to_number = req.phone_number
            if TWILIO_USE_WHATSAPP:
                from_val = TWILIO_FROM if TWILIO_FROM.startswith("whatsapp:") else f"whatsapp:{TWILIO_FROM}"
                to_val   = to_number  if to_number.startswith("whatsapp:")  else f"whatsapp:{to_number}"
            else:
                from_val = TWILIO_FROM
                to_val   = to_number
            client.messages.create(
                body=f"Your Alkhidmat verification code is: {otp}. Valid for 10 minutes.",
                from_=from_val,
                to=to_val
            )
            print(f"[OTP] Sent via Twilio {'WhatsApp' if TWILIO_USE_WHATSAPP else 'SMS'} to {req.phone_number}")
        except ImportError:
            print("[OTP] Twilio library not installed (pip install twilio). Falling back to terminal.")
            print(f"[OTP-DEV] OTP for {req.phone_number}: {otp}")
        except Exception as e:
            print(f"[OTP] Twilio send failed: {e}. Falling back to terminal.")
            print(f"[OTP-DEV] OTP for {req.phone_number}: {otp}")
    else:
        # Development fallback — print to terminal
        print(f"[OTP-DEV] OTP for {req.phone_number}: {otp}")
        print("[OTP-DEV] Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_FROM to enable real delivery.")

    # Never expose OTP in API response (security)
    return {"message": "OTP sent successfully"}


@app.post("/auth/user/firebase-login")
async def firebase_login_endpoint(req: FirebaseLoginRequest):
    """
    Verify Firebase ID token and create/login user.
    Frontend sends the ID token after successful phone auth.
    """
    try:
        # Verify the ID token
        decoded_token = await asyncio.to_thread(firebase_auth.verify_id_token, req.id_token)
        phone_number = decoded_token.get("phone_number")
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="Phone number not found in Firebase token")
        
        # Get or create user in our DB
        # Wrapping in to_thread because these are blocking DB calls
        user = await asyncio.to_thread(get_user_by_phone, phone_number)
        if not user:
            user = await asyncio.to_thread(create_user, phone_number)
            print(f"[AUTH] Created new user for phone: {phone_number}")
        
        # Create new session for this login
        session = await asyncio.to_thread(create_session, user["id"])
        db_session_id = str(session["session_id"])
        
        # Use database session_id as the session token
        session_token = db_session_id
        
        # Store in memory for quick lookup
        user_sessions[session_token] = {
            "user_id": user["id"],
            "db_session_id": db_session_id
        }
        
        # Get all chat history for this user (across all sessions)
        chat_history = await asyncio.to_thread(get_user_chat_history, user["id"])
        
        print(f"[AUTH] Firebase login successful for {phone_number}, session: {db_session_id}")
        
        return {
            "message": "Login successful",
            "session_id": session_token,
            "user": {
                "id": user["id"],
                "phone_number": user["phone_number"],
                "name": user.get("name")
            },
            "chat_history": chat_history
        }
    except Exception as e:
        print(f"[AUTH] Firebase login error: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


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
        "message": "Login successful",
        "session_id": session_token,
        "user": {
            "id": user["id"],
            "phone_number": user["phone_number"],
            "name": user.get("name")
        },
        "chat_history": chat_history
    }

# ── Optional: persist user name ─────────────────────────────────────────────
class UserNameRequest(BaseModel):
    name: str

@app.patch("/auth/user/name")
async def update_user_name(
    req: UserNameRequest,
    session_token: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Update the user's display name."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Session required")
    db_session_id, err = resolve_session_id(session_token)
    if not db_session_id:
        raise HTTPException(status_code=401, detail=err or "Invalid session")
    db_session = get_session(db_session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    user_id = db_session["user_id"]
    # Best-effort update — graceful fail if column doesn't exist
    try:
        from supabase_client import get_supabase_client
        sb = get_supabase_client()
        sb.table("users").update({"name": req.name}).eq("id", str(user_id)).execute()
        print(f"[USER-NAME] Updated name for user {user_id}: '{req.name}'")
    except Exception as e:
        print(f"[USER-NAME] Could not persist name to DB (non-fatal): {e}")
    return {"status": "ok", "name": req.name}


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
    
    # Fetch user info for name
    user_info = get_user_by_id(user_id)
    user_name = user_info.get("name") if user_info else None
    
    # Get all chat history for this user (across all sessions)
    chat_history = get_user_chat_history(user_id)
    
    # ── Check for active tickets across all user sessions ──────────────────
    from supabase_client import get_supabase_client
    supabase = get_supabase_client()
    
    is_agent_chat = False
    active_ticket_id = None
    
    try:
        # Get all sessions for this user to check for active tickets globally
        user_sessions_res = await asyncio.to_thread(
            lambda: supabase.table("sessions").select("session_id").eq("user_id", str(user_id)).execute()
        )
        user_session_ids = [str(s["session_id"]) for s in (user_sessions_res.data or [])]
        
        if user_session_ids:
            # Query tickets that belong to any response in any of the user's sessions
            active_ticket_query = await asyncio.to_thread(
                lambda: supabase.table("tickets")
                .select("ticket_id, status, responses!inner(session_id)")
                .in_("responses.session_id", user_session_ids)
                .in_("status", ["active", "in_progress"])
                .limit(1)
                .execute()
            )
            
            if active_ticket_query.data:
                is_agent_chat = True
                active_ticket_id = str(active_ticket_query.data[0]["ticket_id"])
                print(f"[CREATE-CHAT] User {user_id} has active ticket {active_ticket_id}")
    except Exception as _e:
        print(f"[CREATE-CHAT] Active ticket check error (non-fatal): {_e}")

    return {
        "session_id": session_token,  # Return the token that was sent (could be UUID or old format)
        "db_session_id": str(db_session_id),
        "user_id": str(user_id),
        "user_name": user_name,       # Include the stored name
        "chat_history": chat_history, # Include all previous messages
        "is_agent_chat": is_agent_chat,
        "active_ticket_id": active_ticket_id
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


# @app.post("/chats/{session_id}")
# async def chat(
#     session_id: str, 
#     req: ChatRequest,
#     session_token: Optional[str] = Header(None, alias="X-Session-ID")
# ):

# refined for image processing 
@app.post("/chats/{session_id}")
async def chat(
    session_id: str,
    message: str = Form(""),
    image: UploadFile | None = File(None),
    session_token: Optional[str] = Header(None, alias="X-Session-ID")
):

    """Process chat message with RAG."""
    # Prefer X-Session-ID header if provided, otherwise use URL parameter
    session_to_resolve = session_token or session_id
    
    # Debug logging (opt-in)
    IMAGE_DEBUG = os.getenv("IMAGE_DEBUG", "false").lower() in ("1", "true", "yes", "y")
    if IMAGE_DEBUG:
        print(f"[DEBUG] Chat request - URL session_id: {session_id}, Header X-Session-ID: {session_token}, Resolving: {session_to_resolve}")
    
    db_session_id, error_msg = resolve_session_id(session_to_resolve)
    
    if not db_session_id:
        print(f"[DEBUG] Session resolution failed: {error_msg}")
        raise HTTPException(status_code=401, detail=error_msg or "Invalid or expired session. Please login again.")
    
    # Get or verify session
    db_session = await asyncio.to_thread(get_session, db_session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update session activity
    await asyncio.to_thread(update_session_activity, db_session_id)
    
    # add image_processing step
    final_message = message.strip()
    ocr_text = ""
    if IMAGE_DEBUG:
        print(f"[IMAGE-DEBUG] User text: '{message}'")
        print(f"[IMAGE-DEBUG] Image provided: {image is not None}")
    if image:
        image_bytes = await image.read()
        ocr_text = extract_text_from_image(image_bytes) or ""
        if IMAGE_DEBUG:
            print(f"[IMAGE-DEBUG] OCR result length: {len(ocr_text)} chars")
        if ocr_text:
            final_message += (
                "\n\nText detected in attached image:\n"
                + ocr_text
            )

    
    # Create query record (domain will be updated after RAG processing)
    # Wrapping in to_thread because it involves DB write and possible retries
    query_record = await asyncio.to_thread(
        _db_write_with_retry, 
        create_query, 
        db_session_id, 
        final_message
    )
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
        # ── Optimized Active Ticket Check ──────────────────────────────────────
        from supabase_client import get_supabase_client
        supabase = get_supabase_client()
        
        # Use a single join query to find if this session has an active ticket.
        # Wrapping in to_thread because supabase.execute() is blocking.
        has_active_ticket = False
        active_ticket_id = None
        assigned_agent_id = None
        
        try:
            # Get all sessions for this user to check for active tickets across sessions
            user_id = db_session["user_id"]
            user_sessions_res = await asyncio.to_thread(
                lambda: supabase.table("sessions").select("session_id").eq("user_id", str(user_id)).execute()
            )
            user_session_ids = [str(s["session_id"]) for s in (user_sessions_res.data or [])]
            
            if user_session_ids:
                # Query tickets that belong to any response in any of the user's sessions
                active_ticket_query = await asyncio.to_thread(
                    lambda: supabase.table("tickets")
                    .select("ticket_id, agent_id, status, responses!inner(session_id)")
                    .in_("responses.session_id", user_session_ids)
                    .in_("status", ["active", "in_progress"])
                    .limit(1)
                    .execute()
                )
                
                if active_ticket_query.data:
                    has_active_ticket = True
                    active_ticket_id = str(active_ticket_query.data[0]["ticket_id"])
                    assigned_agent_id = active_ticket_query.data[0].get("agent_id")
                    print(f"[AGENT-CHECK] Found active ticket {active_ticket_id} for user {user_id} across sessions")
        except Exception as _e:
            print(f"[AGENT-CHECK] Optimized check failed: {_e}, falling back to legacy check")
            # Legacy fallback if join fails (schema dependent)
            active_responses = await asyncio.to_thread(
                lambda: supabase.table("responses").select("response_id").eq("session_id", db_session_id).execute()
            )
            response_ids = [r["response_id"] for r in (active_responses.data or [])]
            if response_ids:
                active_tickets = await asyncio.to_thread(
                    lambda: supabase.table("tickets").select("ticket_id, agent_id").in_("response_id", response_ids).in_("status", ["active", "in_progress"]).limit(1).execute()
                )
                if active_tickets.data:
                    has_active_ticket = True
                    active_ticket_id = str(active_tickets.data[0]["ticket_id"])
                    assigned_agent_id = active_tickets.data[0].get("agent_id")
        
        # Check if user explicitly requested human agent
        user_requested_agent = is_human_agent_request(final_message)
        print(f"[HUMAN-AGENT-CHECK] user_requested_agent={user_requested_agent}, has_active_ticket={has_active_ticket}")
        
        # If user requested agent, route immediately WITHOUT RAG processing
        if user_requested_agent:
            print(f"[HUMAN-AGENT-REQUEST] User requested human agent: '{final_message}' - Routing immediately, skipping RAG")
            
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
                try:
                    await asyncio.to_thread(update_query_domain, query_id, query_domain)
                    print(f"[QUERY-DOMAIN] Updated query {query_id} with domain: {query_domain}")
                except Exception as _uqd_err:
                    print(f"[ERROR] Failed to update query domain: {_uqd_err}")
            
            # Create a placeholder response for the ticket
            placeholder_response = await asyncio.to_thread(
                create_response,
                db_session_id,
                f"User requested to chat with human agent: {final_message}",
                confidence=1.0,  # High confidence since it's an explicit request
                domain=query_domain
            )
            # Create ticket immediately with domain for better routing
            ticket = await asyncio.to_thread(create_ticket, placeholder_response["response_id"], domain=query_domain)
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
        
        if has_active_ticket and not user_requested_agent:
            print(f"[AGENT-CHAT] User has active ticket — forwarding message to agent, skipping RAG")
            print(f"[DEBUG-CHAT] Active ticket found: {active_ticket_id} for user: {db_session['user_id']}")
            # Routing to agent
            print(f"[DEBUG-CHAT] Routing user message to agent {assigned_agent_id or 'unassigned'}")
            
            # Broadcast the user's message to the assigned agent's WebSocket
            if active_ticket_id:
                try:
                    msg_payload = {
                        "type": "new_message",
                        "ticket_id": active_ticket_id,
                        "message": {
                            "role": "user",
                            "sender": "user",
                            "content": final_message,
                            "timestamp": datetime.now().isoformat(),
                        }
                    }
                    if assigned_agent_id:
                        await manager.send_to_agent(str(assigned_agent_id), msg_payload)
                    else:
                        # Broadcast to all agents if not yet assigned
                        await manager.broadcast_ticket_update(active_ticket_id, "user_message", msg_payload)
                    print(f"[AGENT-CHAT] Forwarded user message to agent for ticket {active_ticket_id}")
                except Exception as _e:
                    print(f"[AGENT-CHAT] WebSocket forward error (non-fatal): {_e}")

            # Return an ack — the agent will reply via their dashboard
            return {
                "answer": "",          # Empty: agent will respond via WS push
                "sources": [],
                "agent_chat": True,
                "response_id": None,
                "confidence": 1.0,
                "ticket_id": active_ticket_id
            }

        # Process with RAG (use Self-RAG if enabled)
        use_selfrag = getattr(rag_module, 'SELFRAG_ENABLE', False)
        # ── Answer cache: check before running the full RAG pipeline ─────────
        # We embed the English-normalised form of the query (same embedding
        # the retriever uses) and look for a semantically similar cached answer.
        #
        # Cache is skipped when conversation history exists — queries that have
        # been rewritten by memory are session-specific and must not match
        # across users.
        _cache_hit           = None
        _query_en_for_cache  = None
        _embedding_for_cache = None
        _conv_history        = []

        # Cache always operates on the raw user query — never on the
        # memory-rewritten version. Memory rewrites are for retrieval only.
        try:
            from rag_language import build_query_lang_profile
            from rag_embeddings import get_embedder
            _lang_profile        = build_query_lang_profile(final_message)
            _query_en_for_cache  = _lang_profile.query_en
            _embedder            = get_embedder()
            _embedding_for_cache = _embedder.encode(
                [f"query: {_query_en_for_cache}"], normalize_embeddings=True
            )[0].astype("float32")
            _cache_hit = rag_answer_cache.lookup(
                        _query_en_for_cache,
                        _embedding_for_cache,
                        input_lang=_lang_profile.input_lang,
                    )
        except Exception as _ce:
            print(f"[ANSWER-CACHE] Lookup error (non-fatal): {_ce}", flush=True)

        if _cache_hit:
            answer                = _cache_hit["answer"]
            sources               = _cache_hit["sources"]
            confidence_scores     = _cache_hit["confidence_scores"]
            domain_classification = _cache_hit["domain_classification"]
            selfrag_metrics       = _cache_hit.get("selfrag_metrics", {})
            input_lang            = _cache_hit.get("input_lang", "en")
            is_urdu               = input_lang in ("ur", "roman_ur")
            print(f"[ANSWER-CACHE] ✅ Served from cache for: '{final_message[:60]}'", flush=True)
 
        elif use_selfrag:
            # ── Fetch conversation history for memory (Optimized) ─────────────
            try:
                from supabase_client import get_supabase_client as _get_sb
                _sb = _get_sb()
                _q_task = asyncio.to_thread(
                    lambda: _sb.table("queries")
                    .select("content, timestamp")
                    .eq("session_id", str(db_session_id))
                    .order("timestamp", desc=True)
                    .limit(5)
                    .execute()
                )
                _r_task = asyncio.to_thread(
                    lambda: _sb.table("responses")
                    .select("content, timestamp")
                    .eq("session_id", str(db_session_id))
                    .order("timestamp", desc=True)
                    .limit(5)
                    .execute()
                )
                _q_res, _r_res = await asyncio.gather(_q_task, _r_task)
                _q_rows = _q_res.data or []
                _r_rows = _r_res.data or []
                _turns = (
                    [{"role": "user",      "content": r["content"], "ts": r["timestamp"]} for r in _q_rows] +
                    [{"role": "assistant", "content": r["content"], "ts": r["timestamp"]} for r in _r_rows]
                )
                _turns.sort(key=lambda x: x["ts"])
                _conv_history = [{"role": t["role"], "content": t["content"]} for t in _turns]
                print(f"[MEMORY] Loaded {len(_conv_history)} history turns for session {db_session_id}")
            except Exception as _e:
                print(f"[MEMORY] History fetch failed (non-fatal): {_e}")
            # ──────────────────────────────────────────────────────────────────
 
            result = await asyncio.to_thread(
                rag_module.generate_answer_selfrag,
                final_message,
                top_k=5,
                filter_category=None,
                conversation_history=_conv_history,
            )
            answer, _, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics = result
            is_urdu = (input_lang == "ur" or input_lang == "roman_ur")
 
            # ── Store result in answer cache if this was a first-turn query ───
            # First-turn only: no history rewrite risk, answer is self-contained.
            # Store against the original query — memory's rewrite is irrelevant here
            if _embedding_for_cache is not None and _query_en_for_cache:
                try:
                    rag_answer_cache.store(
                        query_en=_query_en_for_cache,
                        embedding=_embedding_for_cache,
                        answer=answer,
                        sources=sources,
                        confidence_scores=confidence_scores,
                        domain_classification=domain_classification,
                        selfrag_metrics=selfrag_metrics,
                        input_lang=input_lang,
                    )
                except Exception as _se:
                    print(f"[ANSWER-CACHE] Store error (non-fatal): {_se}", flush=True)
 
        else:
            # generate_answer returns (answer, query, is_urdu, sources, confidence_scores, domain_classification)
            answer, _, is_urdu, sources, confidence_scores, domain_classification = await asyncio.to_thread(
                rag_module.generate_answer, final_message, top_k=5, filter_category=None
            )
            selfrag_metrics = {}
        
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
            try:
                update_query_domain(query_id, query_domain)
                print(f"[QUERY-DOMAIN] Updated query {query_id} with domain: {query_domain}")
            except Exception as _uqd_err:
                print(f"[ERROR] Failed to update query domain: {_uqd_err}")
        
        # Create response record with confidence (retry on transient network errors)
        response = _db_write_with_retry(
            create_response,
            db_session_id,
            answer,
            confidence=confidence,
            domain=query_domain,
        )
        if response is None:
            # DB write failed — return the answer without a response_id rather than 500ing
            print("[WARNING] create_response failed after retries — returning answer without DB record")
            return convert_numpy_types({
                "answer": answer,
                "sources": sources,
                "agent_chat": False,
                "response_id": None,
                "confidence": confidence,
                "ticket_id": None,
                "ocr_text": ocr_text if ocr_text else None,
                "has_image": bool(image),
            })
        
        # ------------------------------------------------------------------ #
        # TICKET CREATION LOGIC                                               #
        #                                                                     #
        # A ticket is raised when ANY of these is true:                       #
#        1. selfrag_metrics['route_to_agent'] = True (RAG explicitly routed) #
        #    - verify_support returned NO_SUPPORT (hallucination)             #
        #    - check_answer_in_context returned CANNOT_ANSWER                 #
        #    - utility too low and evidence not strong                        #
        #    - no documents found                                             #
        #    - confidence below SELFRAG_MIN_CONFIDENCE                       #
        # 2. confidence < CONFIDENCE_THRESHOLD (retrieval score too low)      #
        #                                                                     #
        # Domain is taken from domain_classification (donation / healthcare / #
        # general) so the ticket is routed to the right department agent.     #
        # ------------------------------------------------------------------ #

        CONFIDENCE_THRESHOLD = float(os.getenv("TICKET_CONFIDENCE_THRESHOLD", "0.70"))
        ticket_created = False
        ticket_id = None
        final_answer = answer  # default; may be replaced below

        # Check if the RAG pipeline explicitly flagged this for agent routing
        rag_routed_to_agent = (
            isinstance(selfrag_metrics, dict) and
            selfrag_metrics.get("route_to_agent", False)
        ) if use_selfrag else False

        # Low-confidence signal (works for both selfrag and non-selfrag paths)
        low_confidence = (
            confidence is not None and confidence < CONFIDENCE_THRESHOLD
        )

        # ── Off-topic domain guard: skip ticket for irrelevant queries ─────
        NON_SERVICE_DOMAINS = {
            "weather", "sports", "entertainment", "cooking", "technology",
            "news", "politics", "geography", "science", "mathematics",
            "history", "music", "movies", "games", "travel"
        }
        def _is_off_topic(domain: Optional[str], msg: str) -> bool:
            if domain and domain.lower().split("/")[0].strip() in NON_SERVICE_DOMAINS:
                return True
            # Keyword-level heuristic for common off-topic queries
            lower_msg = msg.lower()
            off_topic_signals = [
                "weather", "temperature", "forecast", "rain", "sunny",
                "cricket", "football", "match score", "who won",
                "recipe", "cook", "movie", "song", "music",
                "capital of", "president of", "prime minister of",
            ]
            return any(sig in lower_msg for sig in off_topic_signals)

        is_off_topic_query = _is_off_topic(query_domain, final_message)
        if is_off_topic_query:
            print(f"[TICKET] Skipping ticket — off-topic query: domain='{query_domain}', msg='{final_message[:60]}'")

        should_create_ticket = (rag_routed_to_agent or low_confidence) and not has_active_ticket and not is_off_topic_query

        if should_create_ticket:
            ticket_reason = "RAG route_to_agent" if rag_routed_to_agent else f"low confidence ({confidence:.3f})"
            print(f"[TICKET] Creating ticket — reason: {ticket_reason} | domain: {query_domain}", flush=True)

            # Create ticket with the domain from RAG classification so it routes
            # to the right department (donation / healthcare / general)
            ticket = create_ticket(response["response_id"], domain=query_domain)
            if ticket:
                ticket_created = True
                ticket_id = str(ticket.get("ticket_id"))
                agent_assigned = ticket.get("agent_id")
                status = ticket.get("status")
                print(
                    f"[TICKET] Created {ticket_id} "
                    f"(domain: {query_domain}, status: {status}, "
                    f"agent: {agent_assigned})",
                    flush=True
                )

                # Broadcast to agent dashboard via WebSocket
                try:
                    await manager.broadcast_ticket_update(ticket_id, "created", {
                        "ticket_id": ticket_id,
                        "status": status,
                        "agent_id": str(agent_assigned) if agent_assigned else None,
                        "domain": query_domain,
                        "reason": ticket_reason,
                    })
                except Exception as e:
                    print(f"[WS] Error broadcasting ticket creation: {e}")

                # Build a user-facing routing message in the right language
                # Map domain → human-readable department name
                _dept_map = {
                    "donation":   "our Donations team",
                    "healthcare": "our Healthcare team",
                    "general":    "a human agent",
                }
                _dept = _dept_map.get(
                    (query_domain or "").lower().split("/")[0],  # handles 'Donors/General' etc.
                    "a human agent"
                )
                # Also handle DB category names like 'Donors', 'Health'
                if query_domain and query_domain.lower() in ("donors", "donor"):
                    _dept = "our Donations team"
                elif query_domain and query_domain.lower() in ("health", "healthcare"):
                    _dept = "our Healthcare team"

                routing_message_en = (
                    f"I wasn't able to fully answer your question. "
                    f"I am routing you to {_dept} — they will respond shortly."
                )
                # Use input_lang (not is_urdu) so Roman Urdu gets its own message,
                # not an Urdu-script translation.
                if input_lang == "ur":
                    try:
                        from RAG_supabase import translate_english_to_urdu
                        final_answer = translate_english_to_urdu(routing_message_en)
                    except Exception as _te:
                        print(f"[WARNING] Translation failed: {_te}")
                        final_answer = routing_message_en
                elif input_lang == "roman_ur":
                    final_answer = (
                        f"Main aapke sawal ka mukammal jawab nahi de saka. "
                        f"Main aapko {_dept.replace('our ', '').replace(' team', ' team')} se connect kar raha hun — "
                        f"woh jald jawab dein ge."
                    )
                else:
                    final_answer = routing_message_en
        
        # Prepare response and convert all numpy types to native Python types
        response_data = {
            "answer": final_answer,
            "sources": sources if not should_create_ticket else [],  # Don't show sources when routing to agent
            "agent_chat": ticket_created,  # True if ticket was created
            "response_id": str(response["response_id"]),
            "confidence": confidence,
            "ticket_id": ticket_id if ticket_created else None,
            "ocr_text": ocr_text if ocr_text else None,
            "has_image": bool(image)
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
    """List all tickets (for agent dashboard), enriched with user phone number."""
    # Fetch both assigned and unassigned tickets concurrently
    assigned_task = asyncio.to_thread(list_tickets, status=status, agent_id=str(agent["agent_id"]))
    unassigned_task = asyncio.to_thread(list_tickets, status=status, agent_id=None, unassigned_only=True)
    
    assigned, unassigned_list = await asyncio.gather(assigned_task, unassigned_task)
    
    # Combine them for the frontend
    tickets = assigned + unassigned_list

    return {
        "tickets": tickets,
        "assigned": assigned,
        "unassigned": unassigned_list,
        "total": len(tickets)
    }


def _enrich_tickets_with_phone(tickets: list) -> list:
    """Add phone_number to each ticket by looking up user via session."""
    try:
        from supabase_client import get_supabase_client
        supabase = get_supabase_client()
        if not supabase:
            return tickets
        for ticket in tickets:
            try:
                response = ticket.get("response")
                session_id = None
                if isinstance(response, dict):
                    session_id = response.get("session_id")
                elif isinstance(response, list) and response:
                    session_id = response[0].get("session_id")
                if not session_id:
                    ticket["phone_number"] = None
                    continue
                session_row = supabase.table("sessions").select("user_id").eq("session_id", str(session_id)).limit(1).execute()
                user_id = session_row.data[0]["user_id"] if session_row.data else None
                if not user_id:
                    ticket["phone_number"] = None
                    continue
                user_row = supabase.table("users").select("phone_number").eq("id", str(user_id)).limit(1).execute()
                ticket["phone_number"] = user_row.data[0].get("phone_number") if user_row.data else None
            except Exception as _e:
                ticket["phone_number"] = None
    except Exception as _e:
        print(f"[PHONE-ENRICH] Failed (non-fatal): {_e}")
    return tickets


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
    """Mark a ticket as resolved and notify the user to return to AI chat."""
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

    # Notify the user's session so the frontend routes back to AI
    try:
        ticket = get_ticket(ticket_id)
        if ticket:
            response = ticket.get("response")
            session_id = None
            if isinstance(response, dict):
                session_id = response.get("session_id")
            elif isinstance(response, list) and response:
                session_id = response[0].get("session_id")
            if session_id:
                await manager.send_to_user(str(session_id), {
                    "type": "ticket_resolved",
                    "ticket_id": ticket_id,
                    "message": "Your conversation with the agent has ended. You are now back with the AI assistant."
                })
                print(f"[WS] Sent ticket_resolved event to user session {session_id}")
    except Exception as e:
        print(f"[WS] Error sending ticket_resolved to user: {e}")

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


# ============================================================================
# ADMIN SETTINGS ENDPOINTS
# ============================================================================

ADMIN_CONFIG_FILE = Path(".admin_config.json")
DEFAULT_ADMIN_CONFIG = {
    "instruction_prompt": (
        "You are the Alkhidmat Foundation AI assistant. "
        "Help users with information about Al-Khidmat's services including healthcare, education, donations, and orphan care. "
        "Be compassionate, clear, and helpful. Always respond in the same language the user writes in."
    ),
    "guardrails": (
        "1. Only answer questions relevant to Alkhidmat Foundation services.\n"
        "2. Do not provide medical diagnoses or legal advice.\n"
        "3. Always recommend consulting a professional for medical emergencies.\n"
        "4. Do not share personal information of users.\n"
        "5. Keep responses respectful and culturally appropriate."
    )
}

def _load_admin_config() -> dict:
    """Load admin settings from disk, using defaults if not found."""
    try:
        if ADMIN_CONFIG_FILE.exists():
            data = json.loads(ADMIN_CONFIG_FILE.read_text(encoding="utf-8"))
            # Merge with defaults for any missing keys
            return {**DEFAULT_ADMIN_CONFIG, **data}
    except Exception as e:
        print(f"[ADMIN-CONFIG] Error loading config: {e}")
    return dict(DEFAULT_ADMIN_CONFIG)

def _save_admin_config(config: dict):
    """Save admin settings to disk."""
    ADMIN_CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


class AdminSettingsRequest(BaseModel):
    instruction_prompt: Optional[str] = None
    guardrails: Optional[str] = None


@app.get("/admin/settings")
async def get_admin_settings(admin: dict = Depends(get_current_admin)):
    """Get current admin configuration (instruction prompt + guardrails)."""
    return _load_admin_config()


@app.post("/admin/settings")
async def save_admin_settings(req: AdminSettingsRequest, admin: dict = Depends(get_current_admin)):
    """Save admin configuration (instruction prompt and/or guardrails)."""
    config = _load_admin_config()
    if req.instruction_prompt is not None:
        config["instruction_prompt"] = req.instruction_prompt
    if req.guardrails is not None:
        config["guardrails"] = req.guardrails
    _save_admin_config(config)
    print(f"[ADMIN-CONFIG] Settings updated by admin {admin.get('admin_id')}")
    return {"status": "saved", "config": config}


@app.post("/admin/kb/upload")
async def upload_kb_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    admin: dict = Depends(get_current_admin)
):
    """Upload a document to the knowledge base and create embeddings."""
    filename = file.filename
    content = ""
    file_ext = Path(filename).suffix.lower()
    
    # Read file content
    file_bytes = await file.read()
    
    if file_ext == ".txt":
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")
    elif file_ext == ".pdf":
        content = rag_module.extract_text_from_pdf(file_bytes)
    elif file_ext == ".docx":
        content = rag_module.extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document or document is empty")
    
    # Add to KB incrementally
    # We use a virtual path for the database record
    virtual_path = f"Uploaded/{category}/{filename}"
    
    try:
        num_chunks = await asyncio.to_thread(
            rag_module.add_document_incremental,
            file_path=virtual_path,
            content=content,
            category=category,
            filename=filename,
            reindex=True
        )
        
        if num_chunks > 0:
            return {
                "status": "success",
                "filename": filename,
                "category": category,
                "chunks": num_chunks,
                "message": f"Successfully added {filename} to knowledge base"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add document to knowledge base (no chunks created)")
            
    except Exception as e:
        print(f"[ADMIN-KB-UPLOAD] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


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


@app.get("/admin/kb/documents")
async def list_kb_documents(admin: dict = Depends(get_current_admin)):
    """List all unique documents currently in the knowledge base."""
    try:
        docs = await asyncio.to_thread(rag_module.list_knowledge_base_documents)
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        print(f"[ADMIN-KB-LIST] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


class KBDeleteRequest(BaseModel):
    file_path: str


@app.delete("/admin/kb/documents")
async def delete_kb_document(req: KBDeleteRequest, admin: dict = Depends(get_current_admin)):
    """Delete a document and all its embeddings from the knowledge base."""
    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    try:
        success = await asyncio.to_thread(rag_module.delete_document_by_path, req.file_path)
        if success:
            print(f"[ADMIN-KB-DELETE] Admin {admin.get('admin_id')} deleted: {req.file_path}")
            return {"status": "deleted", "file_path": req.file_path}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ADMIN-KB-DELETE] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
        
@app.post("/admin/kb/upload")
async def upload_kb_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    admin: dict = Depends(get_current_admin)
):
    """Upload a document to the knowledge base and create embeddings."""
    filename = file.filename
    content = ""
    file_ext = Path(filename).suffix.lower()
    
    # Read file content
    file_bytes = await file.read()
    
    if file_ext == ".txt":
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")
    elif file_ext == ".pdf":
        content = rag_module.extract_text_from_pdf(file_bytes)
    elif file_ext == ".docx":
        content = rag_module.extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document or document is empty")
    
    # Add to KB incrementally
    # We use a virtual path for the database record
    virtual_path = f"Uploaded/{category}/{filename}"
    
    try:
        num_chunks = await asyncio.to_thread(
            rag_module.add_document_incremental,
            file_path=virtual_path,
            content=content,
            category=category,
            filename=filename,
            reindex=True
        )
        
        if num_chunks > 0:
            return {
                "status": "success",
                "filename": filename,
                "category": category,
                "chunks": num_chunks,
                "message": f"Successfully added {filename} to knowledge base"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add document to knowledge base (no chunks created)")
            
    except Exception as e:
        print(f"[ADMIN-KB-UPLOAD] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# ============================================================================
# EVAL REPORT ENDPOINTS
# ============================================================================

EVAL_REPORT_FILES = {
    "english": "report_english.json",
    "urdu":    "report_urdu.json",
    "roman":   "report_roman.json",
}

@app.get("/admin/eval/status")
async def get_eval_status(admin: dict = Depends(get_current_admin)):
    """Returns which language reports exist on disk."""
    report_dir = Path("evaluation_results")
    status = {}
    for lang, filename in EVAL_REPORT_FILES.items():
        path = report_dir / filename
        if path.exists():
            try:
                stat = path.stat()
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                status[lang] = {
                    "exists": True,
                    "filename": filename,
                    "run_at": data.get("run_at") or datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "language": data.get("language", lang),
                    "total_cases": len(data.get("results", [])),
                }
            except Exception:
                status[lang] = {"exists": True, "filename": filename, "run_at": None, "total_cases": 0}
        else:
            status[lang] = {"exists": False, "filename": filename}
    return status


@app.get("/admin/eval/reports/{language}")
async def get_eval_report_by_language(
    language: str,
    admin: dict = Depends(get_current_admin)
):
    """Returns evaluation results for a given language (english | urdu | roman)."""
    if language not in EVAL_REPORT_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown language '{language}'. Use: english, urdu, roman")

    report_path = Path("evaluation_results") / EVAL_REPORT_FILES[language]
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No report found for '{language}'. Run: python test_rag_evaluation.py --language {language} --use-openai-judge"
        )

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "language": language,
            "run_at": data.get("run_at"),
            "results": data.get("results", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {e}")

@app.post("/admin/cache/clear")
async def clear_answer_cache(admin: dict = Depends(get_current_admin)):
    """
    Flush the entire answer cache.
 
    Call this after rebuilding the knowledge base or changing RAG prompts
    so that stale cached answers are not served to users.
    """
    try:
        cleared = rag_answer_cache.clear()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")
    return {
        "status": "ok",
        "cleared_entries": cleared,
        "message": f"Answer cache cleared. {cleared} entries removed."
    }
 
 
@app.get("/admin/cache/stats")
async def get_cache_stats(admin: dict = Depends(get_current_admin)):
    """
    Return answer cache statistics (total entries, hit count, domain/language breakdown).
    """
    return rag_answer_cache.stats()

# ============================================================================
# LEGACY ENDPOINTS (for backward compatibility)
# ============================================================================

@app.post("/tickets/{ticket_id}/message")
async def send_agent_message(ticket_id: str, req: MessageRequest, agent: dict = Depends(get_current_agent)):
    """Send a message in agent chat. Stores message in database."""
    from supabase_client import get_supabase_client
    
    # Get ticket to find the session (wrapped in to_thread to avoid blocking)
    ticket = await asyncio.to_thread(get_ticket, ticket_id)
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
    
    # Broadcast message to user via WebSocket IMMEDIATELY for speed
    try:
        await manager.send_to_user(str(session_id), {
            "type": "new_message",
            "message": {
                "role": "agent",
                "sender": "agent",
                "content": req.message,
                "timestamp": datetime.now().isoformat(),
                "response_id": None # Will be stored in DB next
            }
        })
        print(f"[WS] Broadcasted agent message to user session {session_id} (pre-storage)")
    except Exception as e:
        print(f"[WS] Error broadcasting agent message: {e}")

    # Store agent message as a response in the database
    # Wrapped in to_thread because DB write is slow/blocking
    agent_response = await asyncio.to_thread(
        create_response,
        session_id,
        req.message,
        confidence=1.0,
        domain=None
    )
    
    print(f"[AGENT-MESSAGE] Agent {agent.get('agent_id')} sent message, stored as response {agent_response.get('response_id') if agent_response else 'FAILED'}")
    
    return {
        "status": "sent",
        "ticket_id": ticket_id,
        "response_id": str(agent_response.get("response_id")) if agent_response else None,
        "message": req.message
    }


@app.get("/tickets/{ticket_id}/chat")
async def get_agent_chat_endpoint(ticket_id: str, agent: dict = Depends(get_current_agent)):
    """Get chat messages for a ticket (legacy endpoint)."""
    from supabase_client import get_supabase_client
    
    ticket = await asyncio.to_thread(get_ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get session ID from ticket response
    response = ticket.get("response")
    if not response:
        raise HTTPException(status_code=404, detail="Response not found for this ticket")
    
    session_id = response.get("session_id") if isinstance(response, dict) else response[0].get("session_id") if isinstance(response, list) else None
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = await asyncio.to_thread(get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = session["user_id"]
    
    # Get all chat history for this user (across all sessions)
    history = await asyncio.to_thread(get_user_chat_history, user_id)
    
    # Convert history to the format expected by the agent dashboard
    # (The dashboard handles the 'role' and 'sender' fields from get_user_chat_history)
    return {
        "ticket_id": ticket_id,
        "messages": history
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