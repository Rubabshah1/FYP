"""
Database operations for Alkhidmat Chat Portal using Supabase client.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import uuid4, UUID
import hashlib
import secrets
import random

from supabase_client import get_supabase_client

# In-memory OTP storage (in production, use Redis)
otp_store: Dict[str, Dict] = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password (simple hash, use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


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
        data["confidence"] = confidence
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
        # Find agents with matching domain who don't have too many active tickets
        agents_result = supabase.table("human_agents").select(
            "agent_id, domain"
        ).eq("domain", domain).execute()
        
        if agents_result.data:
            # Find agent with least active tickets
            best_agent = None
            min_tickets = float('inf')
            
            for agent in agents_result.data:
                # Count active tickets for this agent
                tickets_result = supabase.table("tickets").select(
                    "ticket_id", count="exact"
                ).eq("agent_id", agent["agent_id"]).eq("status", "in_progress").execute()
                
                ticket_count = tickets_result.count if hasattr(tickets_result, 'count') else len(tickets_result.data) if tickets_result.data else 0
                
                if ticket_count < min_tickets:
                    min_tickets = ticket_count
                    best_agent = agent
            
            if best_agent:
                agent_id = best_agent["agent_id"]
    
    ticket_data = {
        "response_id": response_id,
        "status": "active" if not agent_id else "in_progress"  # Auto-assign if agent found
    }
    
    if agent_id:
        ticket_data["agent_id"] = agent_id
    
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
    if stored_password == hashed_input:
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
    if admin["password"] == hash_password(password):
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
    supabase = get_supabase_client()
    
    # Get all tickets with response data
    all_tickets_result = supabase.table("tickets").select("*, responses(confidence, domain, session_id, response_id)").execute()
    all_tickets = all_tickets_result.data or []
    
    # Get all responses to calculate RAG vs Human stats
    all_responses = supabase.table("responses").select("response_id, confidence, domain, timestamp").execute().data or []
    
    # Get ticket response IDs (those that have tickets = human agent handled)
    ticket_response_ids = [t.get("response_id") for t in all_tickets if t.get("response_id")]
    
    # Calculate statistics
    total_tickets = len(all_tickets)
    active_tickets = len([t for t in all_tickets if t.get("status") == "active"])
    in_progress_tickets = len([t for t in all_tickets if t.get("status") == "in_progress"])
    resolved_tickets = len([t for t in all_tickets if t.get("status") == "resolved"])
    
    # RAG vs Human Agent stats
    rag_responses = [r for r in all_responses if r.get("response_id") not in ticket_response_ids]
    human_responses = [r for r in all_responses if r.get("response_id") in ticket_response_ids]
    
    # Calculate average confidence for RAG responses
    rag_confidences = [r.get("confidence") for r in rag_responses if r.get("confidence") is not None]
    avg_rag_confidence = sum(rag_confidences) / len(rag_confidences) if rag_confidences else 0
    
    # Calculate average resolution time for resolved tickets
    resolution_times = []
    for ticket in all_tickets:
        if ticket.get("status") == "resolved" and ticket.get("created_at") and ticket.get("resolved_at"):
            try:
                created = datetime.fromisoformat(ticket["created_at"].replace("Z", "+00:00"))
                resolved = datetime.fromisoformat(ticket["resolved_at"].replace("Z", "+00:00"))
                resolution_time = (resolved - created).total_seconds() / 60  # in minutes
                resolution_times.append(resolution_time)
            except Exception:
                pass
    
    avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    
    # Domain-wise statistics
    domain_stats = {}
    for ticket in all_tickets:
        response = ticket.get("responses")
        if response:
            domain = None
            if isinstance(response, list) and response:
                domain = response[0].get("domain")
            elif isinstance(response, dict):
                domain = response.get("domain")
            
            if domain:
                if domain not in domain_stats:
                    domain_stats[domain] = {"total": 0, "resolved": 0, "active": 0}
                domain_stats[domain]["total"] += 1
                if ticket.get("status") == "resolved":
                    domain_stats[domain]["resolved"] += 1
                elif ticket.get("status") == "active":
                    domain_stats[domain]["active"] += 1
    
    # Time-based statistics (last 7 days, 30 days)
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    last_7_days = now - timedelta(days=7)
    last_30_days = now - timedelta(days=30)
    
    tickets_7d = []
    tickets_30d = []
    for ticket in all_tickets:
        if ticket.get("created_at"):
            try:
                created = datetime.fromisoformat(ticket["created_at"].replace("Z", "+00:00"))
                if created >= last_7_days:
                    tickets_7d.append(ticket)
                if created >= last_30_days:
                    tickets_30d.append(ticket)
            except Exception:
                pass
    
    return {
        "tickets": {
            "total": total_tickets,
            "active": active_tickets,
            "in_progress": in_progress_tickets,
            "resolved": resolved_tickets,
            "last_7_days": len(tickets_7d),
            "last_30_days": len(tickets_30d)
        },
        "rag_vs_human": {
            "rag_responses": len(rag_responses),
            "human_responses": len(human_responses),
            "total_responses": len(all_responses),
            "rag_percentage": round((len(rag_responses) / len(all_responses) * 100) if all_responses else 0, 2),
            "human_percentage": round((len(human_responses) / len(all_responses) * 100) if all_responses else 0, 2),
            "avg_rag_confidence": round(avg_rag_confidence, 3)
        },
        "performance": {
            "avg_resolution_time_minutes": round(avg_resolution_time, 2),
            "resolution_times": resolution_times[:10]  # Last 10 for chart
        },
        "domain_stats": domain_stats,
        "chart_data": {
            "tickets_by_status": {
                "active": active_tickets,
                "in_progress": in_progress_tickets,
                "resolved": resolved_tickets
            },
            "tickets_by_domain": domain_stats,
            "rag_vs_human": {
                "rag": len(rag_responses),
                "human": len(human_responses)
            }
        }
    }
