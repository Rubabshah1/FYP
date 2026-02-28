const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Helper to get auth headers
function getAuthHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const sessionId = localStorage.getItem('user_session_id');
  const agentToken = localStorage.getItem('agent_token');
  const adminToken = localStorage.getItem('admin_token');
  
  if (sessionId) {
    headers['X-Session-ID'] = sessionId;
  }
  if (agentToken) {
    headers['X-Agent-Token'] = agentToken;
  }
  if (adminToken) {
    headers['X-Admin-Token'] = adminToken;
  }
  
  return headers;
}

// ============================================================================
// USER AUTHENTICATION
// ============================================================================

async function sendOTP(phoneNumber) {
  const res = await fetch(`${BASE_URL}/auth/user/send-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phone_number: phoneNumber })
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data;
}

async function verifyOTP(phoneNumber, otp) {
  const res = await fetch(`${BASE_URL}/auth/user/verify-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phone_number: phoneNumber, otp })
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data;
}

// ============================================================================
// AGENT AUTHENTICATION
// ============================================================================

async function agentLogin(email, password) {
  const res = await fetch(`${BASE_URL}/auth/agent/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data;
}

// ============================================================================
// ADMIN AUTHENTICATION
// ============================================================================

async function adminLogin(email, password) {
  const res = await fetch(`${BASE_URL}/auth/admin/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data;
}

// ============================================================================
// CHAT ENDPOINTS
// ============================================================================

async function createChat() {
  const res = await fetch(`${BASE_URL}/chats`, {
    method: 'POST',
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data; // { session_id, db_session_id, user_id, chat_history }
}

async function getChatHistory(sessionId) {
  const res = await fetch(`${BASE_URL}/chats/${sessionId}/history`, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data; // { session_id, messages }
}

async function sendChatMessage(sessionId, message) {
  const res = await fetch(`${BASE_URL}/chats/${sessionId}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ message })
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data; // { answer, sources, agent_chat, response_id }
}
//new function for images
// ✅ ADD THIS NEW FUNCTION BELOW
async function sendChatMessageWithImage(sessionId, formData) {
  const sessionIdFromStorage = localStorage.getItem('user_session_id');
  
  const res = await fetch(`${BASE_URL}/chats/${sessionId}`, {
    method: 'POST',
    headers: {
      'X-Session-ID': sessionIdFromStorage || sessionId,
      // ❌ DO NOT include 'Content-Type' - browser sets it automatically for FormData
    },
    body: formData
  });
  
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data; // { answer, sources, agent_chat, response_id }
}
// ============================================================================
// TICKET ENDPOINTS
// ============================================================================

async function createTicket(sessionId, initialMessage = '') {
  const res = await fetch(`${BASE_URL}/tickets`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ session_id: sessionId, initial_message: initialMessage })
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data; // { ticket_id, status, response_id }
}

// Agent dashboard APIs
async function listTickets(status = null) {
  const params = new URLSearchParams();
  if (status) params.append('status', status);
  const url = `${BASE_URL}/tickets${params.toString() ? '?' + params.toString() : ''}`;
  const res = await fetch(url, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data; // { tickets }
}

async function getTicket(ticketId) {
  const res = await fetch(`${BASE_URL}/tickets/${ticketId}`, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function assignTicket(ticketId) {
  const res = await fetch(`${BASE_URL}/tickets/${ticketId}/assign`, {
    method: 'POST',
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function getTicketChat(ticketId) {
  const res = await fetch(`${BASE_URL}/tickets/${ticketId}/chat`, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function sendAgentMessage(ticketId, message, sender = 'agent') {
  const res = await fetch(`${BASE_URL}/tickets/${ticketId}/message`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ message, sender })
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function resolveTicket(ticketId) {
  const res = await fetch(`${BASE_URL}/tickets/${ticketId}/resolve`, {
    method: 'POST',
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

// ============================================================================
// ADMIN ENDPOINTS
// ============================================================================

async function getAnalytics() {
  const res = await fetch(`${BASE_URL}/admin/analytics`, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function adminListTickets(status = null) {
  const params = new URLSearchParams();
  if (status) params.append('status', status);
  const url = `${BASE_URL}/admin/tickets${params.toString() ? '?' + params.toString() : ''}`;
  const res = await fetch(url, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data; // { tickets }
}

// ============================================================================
// EVAL ENDPOINTS
// ============================================================================

async function getEvalStatus() {
  const res = await fetch(`${BASE_URL}/admin/eval/status`, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data;
}

async function getEvalReportByLanguage(language) {
  const res = await fetch(`${BASE_URL}/admin/eval/reports/${language}`, {
    headers: getAuthHeaders()
  });
  const data = await res.json();
  if (!res.ok) return Promise.reject({ status: res.status, data });
  return data;
}

export default {
  // Authentication
  sendOTP, verifyOTP, agentLogin, adminLogin,
  // Chat
  createChat, sendChatMessage, sendChatMessageWithImage, getChatHistory,
  // Tickets
  createTicket, listTickets, getTicket, assignTicket, getTicketChat, sendAgentMessage, resolveTicket,
  // Admin
  getAnalytics, adminListTickets,
  // Eval
  getEvalStatus, getEvalReportByLanguage
};