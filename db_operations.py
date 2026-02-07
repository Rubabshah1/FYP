"""
Database operations for Alkhidmat Chat Portal using Supabase client.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import uuid4, UUID
import hashlib
import secrets
import random
import numpy as np

from supabase_client import get_supabase_client

# In-memory OTP storage (in production, use Redis)
otp_store: Dict[str, Dict] = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password (simple hash, use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


def to_python_type(value):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()  # Convert numpy scalar to Python native type
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert numpy array to list
    return value


def to_uuid(value) -> str:
    """Convert UUID or string to string UUID."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        try:
            UUID(value)  # Validate
            return value
        except ValueError:
            raise ValueError(f"Invalid UUID format: {value}")
    return str(value)


# ============================================================================
# USER OPERATIONS
# ============================================================================

def create_user(phone_number: str, db=None) -> Optional[Dict]:
    """Create a new user."""
    supabase = get_supabase_client()
    
    # Check if user already exists
    existing = supabase.table("users").select("*").eq("phone_number", phone_number).execute()
    if existing.data:
        return existing.data[0]
    
    # Create new user
    result = supabase.table("users").insert({
        "phone_number": phone_number
    }).execute()
    
    return result.data[0] if result.data else None


def get_user_by_phone(phone_number: str, db=None) -> Optional[Dict]:
    """Get user by phone number."""
    supabase = get_supabase_client()
    result = supabase.table("users").select("*").eq("phone_number", phone_number).execute()
    return result.data[0] if result.data else None


def get_user_by_id(user_id, db=None) -> Optional[Dict]:
    """Get user by ID (accepts UUID or string)."""
    supabase = get_supabase_client()
    user_id = to_uuid(user_id)
    result = supabase.table("users").select("*").eq("id", user_id).execute()
    return result.data[0] if result.data else None


# ============================================================================
# OTP OPERATIONS
# ============================================================================

def generate_otp(phone_number: str) -> str:
    """Generate and store OTP for phone number."""
    otp = str(random.randint(100000, 999999))
    otp_store[phone_number] = {
        "otp": otp,
        "expires_at": datetime.now(timezone.utc).timestamp() + 300  # 5 minutes
    }
    return otp


def verify_otp(phone_number: str, otp: str) -> bool:
    """Verify OTP for phone number."""
    if phone_number not in otp_store:
        return False
    
    stored = otp_store[phone_number]
    if datetime.now(timezone.utc).timestamp() > stored["expires_at"]:
        del otp_store[phone_number]
        return False
    
    if stored["otp"] == otp:
        del otp_store[phone_number]
        return True
    return False


# ============================================================================
# SESSION OPERATIONS
# ============================================================================

def create_session(user_id, db=None) -> Dict:
    """Create a new session for a user."""
    supabase = get_supabase_client()
    user_id = to_uuid(user_id)
    
    result = supabase.table("sessions").insert({
        "user_id": user_id
    }).execute()
    
    return result.data[0] if result.data else None


def get_session(session_id, db=None) -> Optional[Dict]:
    """Get session by ID (accepts UUID or string)."""
    supabase = get_supabase_client()
    session_id = to_uuid(session_id)
    result = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
    return result.data[0] if result.data else None


def update_session_activity(session_id, db=None):
    """Update last_active timestamp for a session."""
    supabase = get_supabase_client()
    session_id = to_uuid(session_id)
    
    supabase.table("sessions").update({
        "last_active": datetime.now(timezone.utc).isoformat()
    }).eq("session_id", session_id).execute()


# ============================================================================
# CHAT HISTORY OPERATIONS
# ============================================================================

def get_user_chat_history(user_id, limit: Optional[int] = None) -> List[Dict]:
    """Get all chat history for a user across all their sessions."""
    supabase = get_supabase_client()
    user_id = to_uuid(user_id)
    
    # Get all sessions for this user
    sessions_result = supabase.table("sessions").select("session_id").eq(
        "user_id", user_id
    ).order("last_active", desc=True).execute()
    
    session_ids = [s["session_id"] for s in (sessions_result.data or [])]
    
    if not session_ids:
        return []
    
    # Get all queries (user messages) from all sessions
    queries_result = supabase.table("queries").select(
        "query_id, session_id, content, timestamp, domain"
    ).in_("session_id", session_ids).order("timestamp", desc=False).execute()
    
    # Get all responses (assistant messages) from all sessions
    responses_result = supabase.table("responses").select(
        "response_id, session_id, content, timestamp, confidence, domain"
    ).in_("session_id", session_ids).order("timestamp", desc=False).execute()
    
    # Combine and sort chronologically
    messages = []
    
    for q in (queries_result.data or []):
        messages.append({
            "role": "user",
            "content": q["content"],
            "timestamp": q["timestamp"],
            "session_id": str(q["session_id"]),
            "query_id": str(q["query_id"]),
            "domain": q.get("domain")
        })
    
    for r in (responses_result.data or []):
        messages.append({
            "role": "assistant",
            "content": r["content"],
            "timestamp": r["timestamp"],
            "session_id": str(r["session_id"]),
            "response_id": str(r["response_id"]),
            "confidence": r.get("confidence"),
            "domain": r.get("domain")
        })
    
    # Sort by timestamp (ascending - oldest first)
    messages.sort(key=lambda x: x["timestamp"])
    
    # Apply limit if specified (return most recent messages)
    if limit:
        messages = messages[-limit:]
    
    return messages


def get_session_chat_history(session_id, limit: Optional[int] = None) -> List[Dict]:
    """Get chat history for a specific session. Identifies agent messages vs RAG responses."""
    supabase = get_supabase_client()
    session_id = to_uuid(session_id)
    
    # Get ticket's initial response_id to identify RAG vs agent messages
    # Agent messages are responses with confidence=1.0 that are NOT the ticket's initial response
    ticket_response_ids = set()
    tickets_result = supabase.table("tickets").select("response_id").execute()
    if tickets_result.data:
        for ticket in tickets_result.data:
            if ticket.get("response_id"):
                ticket_response_ids.add(str(ticket.get("response_id")))
    
    # Get all queries (user messages) for this session
    queries_result = supabase.table("queries").select(
        "query_id, content, timestamp, domain"
    ).eq("session_id", session_id).order("timestamp", desc=False).execute()
    
    # Get all responses (assistant/agent messages) for this session
    responses_result = supabase.table("responses").select(
        "response_id, content, timestamp, confidence, domain"
    ).eq("session_id", session_id).order("timestamp", desc=False).execute()
    
    # Combine and sort chronologically
    messages = []
    
    for q in (queries_result.data or []):
        messages.append({
            "role": "user",
            "content": q["content"],
            "timestamp": q["timestamp"],
            "query_id": str(q["query_id"]),
            "domain": q.get("domain")
        })
    
    for r in (responses_result.data or []):
        response_id = str(r.get("response_id"))
        content = r.get("content", "")
        confidence = r.get("confidence")
        
        # Identify if this is an agent message or RAG response
        # Agent messages: confidence=1.0 AND NOT the ticket's initial response
        # AND not a system/routing message
        is_agent_message = False
        
        if confidence == 1.0:
            # Check if it's NOT the ticket's initial response
            if response_id not in ticket_response_ids:
                # Check if it's NOT a system/routing message
                if (not content.startswith("User requested") and 
                    not content.startswith("I am connecting") and
                    not content.startswith("I don't have enough information")):
                    # This is likely an agent message
                    is_agent_message = True
        
        role = "agent" if is_agent_message else "assistant"
        
        messages.append({
            "role": role,
            "content": content,
            "timestamp": r["timestamp"],
            "response_id": response_id,
            "confidence": confidence,
            "domain": r.get("domain"),
            "sender": "agent" if is_agent_message else "assistant"
        })
    
    # Sort by timestamp (ascending - oldest first)
    messages.sort(key=lambda x: x["timestamp"])
    
    # Apply limit if specified (return most recent messages)
    if limit:
        messages = messages[-limit:]
    
    return messages


# ============================================================================
# QUERY OPERATIONS
# ============================================================================

def create_query(session_id, content: str, domain: Optional[str] = None, db=None) -> Dict:
    """Create a new query."""
    supabase = get_supabase_client()
    session_id = to_uuid(session_id)
    
    data = {
        "session_id": session_id,
        "content": content
    }
    if domain:
        data["domain"] = domain
    
    result = supabase.table("queries").insert(data).execute()
    return result.data[0] if result.data else None


def update_query_domain(query_id, domain: str, db=None) -> bool:
    """Update the domain of an existing query."""
    supabase = get_supabase_client()
    query_id = to_uuid(query_id)
    
    try:
        result = supabase.table("queries").update({
            "domain": domain
        }).eq("query_id", query_id).execute()
        return len(result.data) > 0 if result.data else False
    except Exception as e:
        print(f"[ERROR] Failed to update query domain: {e}")
        return False


# ============================================================================
# RESPONSE OPERATIONS
# ============================================================================

def create_response(
    session_id, 
    content: str, 
    confidence: Optional[float] = None,
    domain: Optional[str] = None,
    db=None
) -> Dict:
    """Create a new response."""
    supabase = get_supabase_client()
    session_id = to_uuid(session_id)
    
    data = {
        "session_id": session_id,
        "content": content
    }
    if confidence is not None:
        # Convert numpy float32/float64 to native Python float for JSON serialization
        data["confidence"] = to_python_type(confidence)
    if domain:
        data["domain"] = domain
    
    result = supabase.table("responses").insert(data).execute()
    return result.data[0] if result.data else None


def link_response_to_documents(response_id, doc_ids: List, db=None):
    """Link response to documents."""
    supabase = get_supabase_client()
    response_id = to_uuid(response_id)
    
    records = []
    for doc_id in doc_ids:
        doc_id = to_uuid(doc_id)
        records.append({
            "response_id": response_id,
            "doc_id": doc_id
        })
    
    if records:
        supabase.table("response_documents").insert(records).execute()


# ============================================================================
# TICKET OPERATIONS
# ============================================================================

def create_ticket(response_id, domain: Optional[str] = None, db=None) -> Dict:
    """Create a new ticket. Optionally route to domain-specific agent."""
    supabase = get_supabase_client()
    response_id = to_uuid(response_id)
    
    # If domain is provided, try to find an available agent for that domain
    agent_id = None
    if domain:
        # Normalize domain (lowercase, strip whitespace)
        domain_normalized = domain.lower().strip() if domain else None
        print(f"[CREATE-TICKET] Looking for agents with domain: '{domain}' (normalized: '{domain_normalized}')")
        
        # First, get all agents to see what domains exist
        all_agents_result = supabase.table("human_agents").select("agent_id, domain, email").execute()
        all_agents = all_agents_result.data or []
        print(f"[CREATE-TICKET] Total agents in database: {len(all_agents)}")
        if all_agents:
            print(f"[CREATE-TICKET] Available agent domains: {[a.get('domain') for a in all_agents]}")
        
        # Find agents with matching domain (case-insensitive)
        # Try exact match first
        agents_result = supabase.table("human_agents").select(
            "agent_id, domain, email"
        ).eq("domain", domain).execute()
        
        # If no exact match, try case-insensitive search
        if not agents_result.data or len(agents_result.data) == 0:
            print(f"[CREATE-TICKET] No exact match for domain '{domain}', trying case-insensitive search...")
            # Get all agents and filter manually (Supabase doesn't support case-insensitive eq)
            agents_result = supabase.table("human_agents").select(
                "agent_id, domain, email"
            ).execute()
            if agents_result.data:
                matching_agents = [
                    a for a in agents_result.data 
                    if a.get("domain") and a.get("domain").lower().strip() == domain_normalized
                ]
                agents_result.data = matching_agents
        
        if agents_result.data and len(agents_result.data) > 0:
            print(f"[CREATE-TICKET] Found {len(agents_result.data)} agent(s) with domain '{domain}': {[a.get('email') for a in agents_result.data]}")
            # Find agent with least active tickets
            best_agent = None
            min_tickets = float('inf')
            
            for agent in agents_result.data:
                # Count active tickets for this agent
                tickets_result = supabase.table("tickets").select(
                    "ticket_id", count="exact"
                ).eq("agent_id", agent["agent_id"]).eq("status", "in_progress").execute()
                
                ticket_count = tickets_result.count if hasattr(tickets_result, 'count') else len(tickets_result.data) if tickets_result.data else 0
                print(f"[CREATE-TICKET] Agent {agent.get('email')} has {ticket_count} in_progress tickets")
                
                if ticket_count < min_tickets:
                    min_tickets = ticket_count
                    best_agent = agent
            
            if best_agent:
                agent_id = best_agent["agent_id"]
                print(f"[CREATE-TICKET] Selected agent {best_agent.get('email')} (agent_id: {agent_id}) with {min_tickets} tickets")
            else:
                print(f"[CREATE-TICKET] No best agent found (this shouldn't happen)")
        else:
            print(f"[CREATE-TICKET] No agents found with domain '{domain}' - ticket will be unassigned")
    
    ticket_data = {
        "response_id": response_id,
        "status": "active" if not agent_id else "in_progress"  # Auto-assign if agent found
    }
    
    if agent_id:
        ticket_data["agent_id"] = agent_id
        print(f"[CREATE-TICKET] Ticket will be auto-assigned to agent {agent_id} with status 'in_progress'")
    else:
        print(f"[CREATE-TICKET] Ticket will be created with status 'active' (unassigned)")
    
    result = supabase.table("tickets").insert(ticket_data).execute()
    
    return result.data[0] if result.data else None


def get_ticket(ticket_id, db=None) -> Optional[Dict]:
    """Get ticket by ID (accepts UUID or string)."""
    supabase = get_supabase_client()
    ticket_id = to_uuid(ticket_id)
    
    # Get ticket with related data
    result = supabase.table("tickets").select("*, responses(*, sessions(*))").eq("ticket_id", ticket_id).execute()
    ticket = result.data[0] if result.data else None
    
    if ticket and ticket.get("responses"):
        # Flatten the response data
        response = ticket["responses"][0] if isinstance(ticket["responses"], list) else ticket["responses"]
        ticket["response"] = response
        if response and response.get("sessions"):
            session = response["sessions"][0] if isinstance(response["sessions"], list) else response["sessions"]
            ticket["session"] = session
    
    return ticket


def list_tickets(
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
    unassigned_only: bool = False,
    db=None
) -> List[Dict]:
    """List tickets with optional filters.
    
    Args:
        status: Filter by status (active, in_progress, resolved)
        agent_id: Filter by assigned agent
        unassigned_only: If True, only return unassigned tickets (status=active, agent_id=null)
    """
    supabase = get_supabase_client()
    
    query = supabase.table("tickets").select("*, responses(content, session_id, domain, confidence)")
    
    if unassigned_only:
        # Show only unassigned tickets (active status, no agent assigned)
        query = query.eq("status", "active").is_("agent_id", "null")
    else:
        if status:
            query = query.eq("status", status)
        
        if agent_id:
            agent_id = to_uuid(agent_id)
            query = query.eq("agent_id", agent_id)
    
    result = query.order("created_at", desc=True).execute()
    tickets = result.data or []
    
    # Format the response
    formatted_tickets = []
    for ticket in tickets:
        formatted = {
            "ticket_id": ticket["ticket_id"],
            "response_id": ticket["response_id"],
            "status": ticket["status"],
            "agent_id": ticket.get("agent_id"),
            "created_at": ticket.get("created_at"),
            "resolved_at": ticket.get("resolved_at"),
            "is_assigned": ticket.get("agent_id") is not None,
            "response": None
        }
        
        # Handle response data
        if ticket.get("responses"):
            response = ticket["responses"][0] if isinstance(ticket["responses"], list) else ticket["responses"]
            formatted["response"] = {
                "content": response.get("content"),
                "session_id": response.get("session_id"),
                "domain": response.get("domain"),
                "confidence": response.get("confidence")
            }
        
        formatted_tickets.append(formatted)
    
    return formatted_tickets


def assign_ticket(ticket_id, agent_id, db=None) -> bool:
    """Assign a ticket to an agent."""
    supabase = get_supabase_client()
    ticket_id = to_uuid(ticket_id)
    agent_id = to_uuid(agent_id)
    
    try:
        supabase.table("tickets").update({
            "agent_id": agent_id,
            "status": "in_progress"
        }).eq("ticket_id", ticket_id).execute()
        return True
    except Exception:
        return False


def resolve_ticket(ticket_id, db=None) -> bool:
    """Mark a ticket as resolved."""
    supabase = get_supabase_client()
    ticket_id = to_uuid(ticket_id)
    
    try:
        supabase.table("tickets").update({
            "status": "resolved",
            "resolved_at": datetime.now(timezone.utc).isoformat()
        }).eq("ticket_id", ticket_id).execute()
        return True
    except Exception:
        return False


# ============================================================================
# AGENT OPERATIONS
# ============================================================================

def create_agent(
    first_name: str,
    last_name: str,
    email: str,
    password: str,
    domain: Optional[str] = None,
    db=None
) -> Dict:
    """Create a new agent."""
    supabase = get_supabase_client()
    
    result = supabase.table("human_agents").insert({
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": hash_password(password),
        "domain": domain
    }).execute()
    
    return result.data[0] if result.data else None


def authenticate_agent(email: str, password: str, db=None) -> Optional[Dict]:
    """Authenticate agent."""
    supabase = get_supabase_client()
    
    # Normalize email (lowercase, strip whitespace)
    email = email.lower().strip()
    password = password.strip()
    
    result = supabase.table("human_agents").select("*").eq("email", email).execute()
    if not result.data:
        return None
    
    agent = result.data[0]
    hashed_input = hash_password(password)
    stored_password = agent.get("password")
    print(f"Stored password: {stored_password}")
    print(f"Hashed input: {hashed_input}")
    # if stored_password == hashed_input:
    if stored_password == password:
        return agent
    return None


def get_agent(agent_id, db=None) -> Optional[Dict]:
    """Get agent by ID (accepts UUID or string)."""
    supabase = get_supabase_client()
    agent_id = to_uuid(agent_id)
    result = supabase.table("human_agents").select("*").eq("agent_id", agent_id).execute()
    return result.data[0] if result.data else None


# ============================================================================
# ADMIN OPERATIONS
# ============================================================================

def create_admin(
    first_name: str,
    last_name: str,
    email: str,
    password: str,
    db=None
) -> Dict:
    """Create a new admin."""
    supabase = get_supabase_client()
    
    result = supabase.table("admins").insert({
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": hash_password(password)
    }).execute()
    
    return result.data[0] if result.data else None


def authenticate_admin(email: str, password: str, db=None) -> Optional[Dict]:
    """Authenticate admin."""
    supabase = get_supabase_client()
    
    # Normalize email (lowercase, strip whitespace)
    email = email.lower().strip()
    password = password.strip()
    
    result = supabase.table("admins").select("*").eq("email", email).execute()
    if not result.data:
        return None
    print(f"Stored password: {result.data[0]['password']}")
    print(f"Hashed input: {hash_password(password)}")
    admin = result.data[0]
    # if admin["password"] == hash_password(password):
    if admin["password"] == (password):
        return admin
    return None


def get_admin(admin_id, db=None) -> Optional[Dict]:
    """Get admin by ID (accepts UUID or string)."""
    supabase = get_supabase_client()
    admin_id = to_uuid(admin_id)
    result = supabase.table("admins").select("*").eq("admin_id", admin_id).execute()
    return result.data[0] if result.data else None


# ============================================================================
# ANALYTICS OPERATIONS
# ============================================================================

def get_ticket_analytics(db=None) -> Dict:
    """Get comprehensive analytics for admin dashboard."""
    try:
        supabase = get_supabase_client()
        
        # Get all queries (user messages)
        all_queries_result = supabase.table("queries").select("query_id, timestamp, domain").execute()
        all_queries = all_queries_result.data or []
        
        # Get all tickets with response data
        all_tickets_result = supabase.table("tickets").select("*, responses(confidence, domain, session_id, response_id)").execute()
        all_tickets = all_tickets_result.data or []
        
        # Get all responses to calculate RAG vs Human stats
        all_responses = supabase.table("responses").select("response_id, confidence, domain, timestamp").execute().data or []
    except Exception as e:
        print(f"[ERROR] Failed to fetch analytics data: {e}")
        # Return empty analytics structure
        return {
            "total_queries": 0,
            "total_rag_answered": 0,
            "total_human_answered": 0,
            "total_tickets": 0,
            "active_tickets": 0,
            "in_progress_tickets": 0,
            "resolved_tickets": 0,
            "average_resolution_time_seconds": 0,
            "queries_over_time": {
                "daily": [],
                "monthly": [],
                "yearly": []
            },
            "rag_vs_human": {
                "rag_responses": 0,
                "human_responses": 0,
                "rag_percentage": 0,
                "human_percentage": 0,
                "avg_rag_confidence": 0
            }
        }
    
    # Get ticket response IDs (those that have tickets = human agent handled)
    ticket_response_ids = set([str(t.get("response_id")) for t in all_tickets if t.get("response_id")])
    
    # Calculate statistics
    total_queries = len(all_queries)
    total_tickets = len(all_tickets)
    active_tickets = len([t for t in all_tickets if t.get("status") == "active"])
    in_progress_tickets = len([t for t in all_tickets if t.get("status") == "in_progress"])
    resolved_tickets = len([t for t in all_tickets if t.get("status") == "resolved"])
    
    # RAG vs Human Agent stats
    rag_responses = [r for r in all_responses if str(r.get("response_id")) not in ticket_response_ids]
    human_responses = [r for r in all_responses if str(r.get("response_id")) in ticket_response_ids]
    
    total_rag_answered = len(rag_responses)
    total_human_answered = len(human_responses)
    
    # Calculate average confidence for RAG responses
    rag_confidences = [r.get("confidence") for r in rag_responses if r.get("confidence") is not None]
    avg_rag_confidence = sum(rag_confidences) / len(rag_confidences) if rag_confidences else 0
    
    # Calculate average resolution time for resolved tickets (in seconds)
    resolution_times = []
    for ticket in all_tickets:
        if ticket.get("status") == "resolved" and ticket.get("created_at") and ticket.get("resolved_at"):
            try:
                created = datetime.fromisoformat(ticket["created_at"].replace("Z", "+00:00"))
                resolved = datetime.fromisoformat(ticket["resolved_at"].replace("Z", "+00:00"))
                resolution_time = (resolved - created).total_seconds()  # in seconds
                resolution_times.append(resolution_time)
            except Exception:
                pass
    
    avg_resolution_time_seconds = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    
    # Time-based statistics - queries over time
    from datetime import timedelta
    from collections import defaultdict
    
    now = datetime.now(timezone.utc)
    
    # Daily data (last 30 days)
    daily_data = defaultdict(lambda: {"total": 0, "rag": 0, "human": 0})
    for query in all_queries:
        if query.get("timestamp"):
            try:
                query_time = datetime.fromisoformat(query["timestamp"].replace("Z", "+00:00"))
                date_key = query_time.strftime("%Y-%m-%d")
                daily_data[date_key]["total"] += 1
                
                # Find corresponding response to determine if RAG or human
                # This is approximate - we check if there's a ticket for responses around this time
                # For simplicity, we'll count based on whether there's a ticket
                # A more accurate approach would require joining queries with responses
            except Exception:
                pass
    
    # Count RAG vs Human for each day based on responses
    for response in all_responses:
        if response.get("timestamp"):
            try:
                response_time = datetime.fromisoformat(response["timestamp"].replace("Z", "+00:00"))
                date_key = response_time.strftime("%Y-%m-%d")
                response_id = str(response.get("response_id"))
                if response_id in ticket_response_ids:
                    daily_data[date_key]["human"] += 1
                else:
                    daily_data[date_key]["rag"] += 1
            except Exception:
                pass
    
    # Convert to list format for chart
    daily_list = [{"date": date, **data} for date, data in sorted(daily_data.items())]
    
    # Monthly data (last 12 months)
    monthly_data = defaultdict(lambda: {"total": 0, "rag": 0, "human": 0})
    for query in all_queries:
        if query.get("timestamp"):
            try:
                query_time = datetime.fromisoformat(query["timestamp"].replace("Z", "+00:00"))
                month_key = query_time.strftime("%Y-%m")
                monthly_data[month_key]["total"] += 1
            except Exception:
                pass
    
    for response in all_responses:
        if response.get("timestamp"):
            try:
                response_time = datetime.fromisoformat(response["timestamp"].replace("Z", "+00:00"))
                month_key = response_time.strftime("%Y-%m")
                response_id = str(response.get("response_id"))
                if response_id in ticket_response_ids:
                    monthly_data[month_key]["human"] += 1
                else:
                    monthly_data[month_key]["rag"] += 1
            except Exception:
                pass
    
    monthly_list = [{"month": month, **data} for month, data in sorted(monthly_data.items())]
    
    # Yearly data (all years)
    yearly_data = defaultdict(lambda: {"total": 0, "rag": 0, "human": 0})
    for query in all_queries:
        if query.get("timestamp"):
            try:
                query_time = datetime.fromisoformat(query["timestamp"].replace("Z", "+00:00"))
                year_key = query_time.strftime("%Y")
                yearly_data[year_key]["total"] += 1
            except Exception:
                pass
    
    for response in all_responses:
        if response.get("timestamp"):
            try:
                response_time = datetime.fromisoformat(response["timestamp"].replace("Z", "+00:00"))
                year_key = response_time.strftime("%Y")
                response_id = str(response.get("response_id"))
                if response_id in ticket_response_ids:
                    yearly_data[year_key]["human"] += 1
                else:
                    yearly_data[year_key]["rag"] += 1
            except Exception:
                pass
    
    yearly_list = [{"year": year, **data} for year, data in sorted(yearly_data.items())]
    
    # Return structure matching frontend expectations
    return {
        "total_queries": total_queries,
        "total_rag_answered": total_rag_answered,
        "total_human_answered": total_human_answered,
        "total_tickets": total_tickets,
        "active_tickets": active_tickets,
        "in_progress_tickets": in_progress_tickets,
        "resolved_tickets": resolved_tickets,
        "average_resolution_time_seconds": round(avg_resolution_time_seconds, 2),
        "queries_over_time": {
            "daily": daily_list,
            "monthly": monthly_list,
            "yearly": yearly_list
        },
        "rag_vs_human": {
            "rag_responses": total_rag_answered,
            "human_responses": total_human_answered,
            "rag_percentage": round((total_rag_answered / total_queries * 100) if total_queries > 0 else 0, 2),
            "human_percentage": round((total_human_answered / total_queries * 100) if total_queries > 0 else 0, 2),
            "avg_rag_confidence": round(avg_rag_confidence, 3)
        }
    }