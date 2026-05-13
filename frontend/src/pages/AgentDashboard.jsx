import { useState, useEffect, useLayoutEffect, useRef, useMemo, useCallback } from 'react';
import { useImmer } from 'use-immer';
import api from '@/api';
import logo from '@/assets/images/alkhidmat.png';
import Spinner from '@/components/Spinner';
import { useWebSocket } from '@/hooks/useWebSocket';

function AgentDashboard() {
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [chatMessages, setChatMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  const [filter, setFilter] = useState('active'); // active, in_progress, resolved
  const [loadingChat, setLoadingChat] = useState(false); // Loading state for chat messages
  const loadingRef = useRef(false); // Prevent concurrent requests
  const chatCacheRef = useRef(new Map()); // Cache for chat messages by ticket_id
  const chatMessagesEndRef = useRef(null); // Ref for scrolling to bottom
  const chatContainerRef = useRef(null); // Ref for the scrollable chat container
  
  // Get agent token from localStorage
  const agentToken = typeof window !== 'undefined' ? localStorage.getItem('agent_token') : null;

  // Load tickets with request deduplication - loads ALL tickets (no filter)
  const loadTickets = useCallback(async (force = false) => {
    // Allow force refresh even if already loading
    if (!force && loadingRef.current) {
      console.log('[AgentDashboard] Already loading tickets, skipping...');
      return; // Skip if already loading (unless forced)
    }
    loadingRef.current = true;
    try {
      console.log('[AgentDashboard] Loading all tickets', force ? '(forced)' : '');
      const response = await api.listTickets(null); // Load all tickets, filter locally
      console.log('[AgentDashboard] Received tickets:', response);
      const ticketsData = response?.tickets || [];
      console.log('[AgentDashboard] Setting tickets:', ticketsData.length, 'tickets');
      setTickets(ticketsData);
    } catch (err) {
      console.error('[AgentDashboard] Failed to load tickets:', err);
    } finally {
      loadingRef.current = false;
    }
  }, []); // No dependencies - stable function reference

  // Load chat messages with caching
  const loadChatMessages = useCallback(async (forceRefresh = false, silent = false) => {
    if (!selectedTicket || loadingRef.current) return;
    
    const ticketId = selectedTicket.ticket_id;
    
    // Check cache first (unless force refresh)
    if (!forceRefresh && chatCacheRef.current.has(ticketId)) {
      const cachedMessages = chatCacheRef.current.get(ticketId);
      console.log('[AgentDashboard] Loading from cache for ticket:', ticketId, cachedMessages.length, 'messages');
      setChatMessages(cachedMessages);
      // Optionally refresh in background without showing loading
      loadingRef.current = true;
      try {
        const chat = await api.getTicketChat(ticketId);
        const messages = chat.messages || [];
        // Update cache and messages if they changed
        if (JSON.stringify(messages) !== JSON.stringify(cachedMessages)) {
          chatCacheRef.current.set(ticketId, messages);
          setChatMessages(messages);
        }
      } catch (err) {
        console.error('[AgentDashboard] Failed to refresh chat messages:', err);
        // Keep cached version on error
      } finally {
        loadingRef.current = false;
      }
      return;
    }
    
    // Fetch from API if not in cache or force refresh
    loadingRef.current = true;
    if (!silent) setLoadingChat(true);
    try {
      console.log('[AgentDashboard] Loading chat messages for ticket:', ticketId);
      const chat = await api.getTicketChat(ticketId);
      console.log('[AgentDashboard] Received chat messages:', chat);
      const messages = chat.messages || [];
      console.log('[AgentDashboard] Setting chat messages:', messages.length, 'messages');
      // Update cache
      chatCacheRef.current.set(ticketId, messages);
      setChatMessages(messages);
    } catch (err) {
      console.error('[AgentDashboard] Failed to load chat messages:', err);
    } finally {
      loadingRef.current = false;
      setLoadingChat(false);
    }
  }, [selectedTicket, setChatMessages]);

  // WebSocket connection for real-time ticket updates
  const handleWebSocketMessage = useCallback((data) => {
    console.log('[AgentDashboard] WebSocket message received:', data);
    
    // 1. Handle Ticket Status Updates (Created, Assigned, Resolved)
    if (data.type === 'ticket_update' && data.update_type !== 'user_message') {
      // Reload tickets when ticket is created, assigned, or resolved
      loadTickets(true);
      
      // If the updated ticket is currently selected, refresh its chat
      if (selectedTicket && selectedTicket.ticket_id === data.ticket_id) {
        loadChatMessages(true, true);
      }
    }
    
    // 2. Handle Real-time User Messages
    if (data.type === 'new_message' || (data.type === 'ticket_update' && data.update_type === 'user_message')) {
      const messageData = data.type === 'new_message' ? data.message : data.data.message;
      const tId = data.ticket_id;
      
      if (!messageData) return;

      console.log('[AgentDashboard] Processing real-time message for ticket:', tId);
      
      // If the message is for the currently selected ticket, update the chat view
      if (selectedTicket && selectedTicket.ticket_id === tId) {
        setChatMessages(draft => {
          // Prevent duplicates (simple content + timestamp check)
          const isDuplicate = draft.some(msg => 
            msg.content === messageData.content && 
            Math.abs(new Date(msg.timestamp) - new Date(messageData.timestamp)) < 2000
          );
          
          if (!isDuplicate) {
            draft.push({
              ...messageData,
              sender: messageData.sender || 'user',
              role: messageData.role || 'user'
            });
          }
        });
      }
      
      // Update the ticket list to show the new message as a preview
      setTickets(prevTickets => 
        prevTickets.map(ticket => 
          ticket.ticket_id === tId 
            ? { 
                ...ticket, 
                response: { 
                  ...ticket.response, 
                  content: messageData.content 
                } 
              } 
            : ticket
        )
      );
    }
  }, [loadTickets, selectedTicket, loadChatMessages, setChatMessages, setTickets]);

  const handleWebSocketError = useCallback((error) => {
    console.error('[AgentDashboard] WebSocket error:', error);
  }, []);

  const handleWebSocketOpen = useCallback(() => {
    console.log('[AgentDashboard] WebSocket connected');
  }, []);

  const handleWebSocketClose = useCallback((event) => {
    console.log('[AgentDashboard] WebSocket closed:', event.code, event.reason);
  }, []);

  const { connected: wsConnected } = useWebSocket(
    agentToken ? `/ws/agent/${agentToken}` : null,
    {
      enabled: !!agentToken,
      onMessage: handleWebSocketMessage,
      onError: handleWebSocketError,
      onOpen: handleWebSocketOpen,
      onClose: handleWebSocketClose
    }
  );

  // Load tickets on mount and when page becomes visible (refresh, tab switch)
  useEffect(() => {
    // Load tickets immediately on mount (force refresh on page load)
    loadTickets(true);
    
    // Also reload tickets when page becomes visible (user switches back to tab or refreshes)
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        console.log('[AgentDashboard] Page became visible, reloading tickets');
        loadTickets(true); // Force refresh when page becomes visible
      }
    };
    
    // Listen for visibility changes
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Also listen for focus events (when user switches back to tab)
    const handleFocus = () => {
      console.log('[AgentDashboard] Window focused, reloading tickets');
      loadTickets(true); // Force refresh on focus
    };
    window.addEventListener('focus', handleFocus);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', handleFocus);
    };
  }, [loadTickets]);

  // Scroll to bottom of chat when messages change
  const scrollToBottom = useCallback(() => {
    if (chatMessagesEndRef.current) {
      chatMessagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    } else if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, []);

  // Scroll to bottom immediately when messages are loaded (useLayoutEffect for instant scroll)
  useLayoutEffect(() => {
    if (chatMessages.length > 0 && !loadingChat) {
      // Use setTimeout to ensure DOM is updated
      setTimeout(() => {
        if (chatMessagesEndRef.current) {
          chatMessagesEndRef.current.scrollIntoView({ behavior: 'auto' });
        } else if (chatContainerRef.current) {
          chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
      }, 0);
    }
  }, [chatMessages.length, loadingChat]);

  // Load chat messages only when ticket is selected
  useEffect(() => {
    if (selectedTicket) {
      const ticketId = selectedTicket.ticket_id;
      // Check cache first - if available, load immediately without loading state
      if (chatCacheRef.current.has(ticketId)) {
        const cachedMessages = chatCacheRef.current.get(ticketId);
        console.log('[AgentDashboard] Loading from cache for ticket:', ticketId);
        setChatMessages(cachedMessages);
        // Refresh in background without showing loading
        if (!loadingRef.current) {
          loadingRef.current = true;
          api.getTicketChat(ticketId)
            .then(chat => {
              const messages = chat.messages || [];
              // Update cache and messages if they changed
              if (JSON.stringify(messages) !== JSON.stringify(cachedMessages)) {
                chatCacheRef.current.set(ticketId, messages);
                setChatMessages(messages);
              }
            })
            .catch(err => {
              console.error('[AgentDashboard] Failed to refresh chat messages:', err);
            })
            .finally(() => {
              loadingRef.current = false;
            });
        }
      } else {
        // Not in cache, show loading and fetch
        setChatMessages([]);
        loadChatMessages(true);
      }
    } else {
      setChatMessages([]);
      setLoadingChat(false);
    }
  }, [selectedTicket, loadChatMessages, setChatMessages]);


  async function assignTicket(ticketId) {
    // Optimistically update local state immediately
    const agentId = localStorage.getItem('agent_id');
    setTickets(prevTickets => 
      prevTickets.map(ticket => 
        ticket.ticket_id === ticketId 
          ? { ...ticket, status: 'in_progress', agent_id: agentId }
          : ticket
      )
    );
    
    // Update selected ticket if it's the one being assigned
    if (selectedTicket?.ticket_id === ticketId) {
      setSelectedTicket(prev => prev ? { ...prev, status: 'in_progress', agent_id: agentId } : null);
    }

    try {
      await api.assignTicket(ticketId);
      // Fetch fresh data from server to sync state (load all tickets)
      const response = await api.listTickets(null);
      const ticketsData = response?.tickets || [];
      setTickets(ticketsData);
      // Update selected ticket with fresh data if it's the one we assigned
      if (selectedTicket?.ticket_id === ticketId) {
        const updatedTicket = ticketsData.find(t => t.ticket_id === ticketId);
        if (updatedTicket) {
          setSelectedTicket(updatedTicket);
        }
      }
    } catch (err) {
      console.error('Failed to assign ticket:', err);
      // Revert optimistic update on error - fetch fresh data
      await loadTickets();
    }
  }

  async function resolveTicket(ticketId) {
    // Optimistically update local state immediately
    const now = new Date().toISOString();
    setTickets(prevTickets => 
      prevTickets.map(ticket => 
        ticket.ticket_id === ticketId 
          ? { ...ticket, status: 'resolved', resolved_at: now }
          : ticket
      )
    );
    
    // Update selected ticket if it's the one being resolved
    if (selectedTicket?.ticket_id === ticketId) {
      setSelectedTicket(prev => prev ? { ...prev, status: 'resolved', resolved_at: now } : null);
    }

    try {
      await api.resolveTicket(ticketId);
      // Fetch fresh data from server to sync state
      await loadTickets();
      // Clear selected ticket if it was resolved
      if (selectedTicket?.ticket_id === ticketId) {
        setSelectedTicket(null);
        setChatMessages([]);
      }
    } catch (err) {
      console.error('Failed to resolve ticket:', err);
      // Revert optimistic update on error
      await loadTickets();
    }
  }

  async function sendMessage() {
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || !selectedTicket) return;

    const ticketId = selectedTicket.ticket_id;

    // Optimistically update UI immediately
    const optimisticMessage = {
      sender: 'agent',
      content: trimmedMessage,
      timestamp: new Date().toISOString()
    };
    setChatMessages(draft => {
      const updated = [...draft, optimisticMessage];
      // Update cache immediately
      chatCacheRef.current.set(ticketId, updated);
      return updated;
    });
    setNewMessage('');

    // Scroll to bottom after adding message
    setTimeout(() => scrollToBottom(), 100);

    try {
      const result = await api.sendAgentMessage(ticketId, trimmedMessage, 'agent');
      console.log('[AgentDashboard] Message sent successfully:', result);
      
      // Update optimistic message with server response data if available
      setChatMessages(draft => {
        const updated = draft.map(msg => 
          msg === optimisticMessage 
            ? { ...msg, response_id: result.response_id, timestamp: new Date().toISOString() }
            : msg
        );
        // Update cache
        chatCacheRef.current.set(ticketId, updated);
        return updated;
      });
      
      // Refresh chat messages after a short delay to get the latest from server
      // This ensures the message is properly stored and synced
      setTimeout(async () => {
        await loadChatMessages(true, true);
      }, 500);
    } catch (err) {
      console.error('Failed to send message:', err);
      // Revert optimistic update on error
      setChatMessages(draft => {
        const reverted = draft.filter(msg => msg !== optimisticMessage);
        // Update cache with reverted state
        chatCacheRef.current.set(ticketId, reverted);
        return reverted;
      });
    }
  }

  async function selectTicket(ticket) {
    // If ticket is active, assign it optimistically first, then select it
    if (ticket.status === 'active') {
      // Optimistically update ticket status and select it immediately
      const agentId = localStorage.getItem('agent_id');
      const optimisticTicket = { ...ticket, status: 'in_progress', agent_id: agentId };
      setTickets(prevTickets => 
        prevTickets.map(t => 
          t.ticket_id === ticket.ticket_id ? optimisticTicket : t
        )
      );
      setSelectedTicket(optimisticTicket);
      
      // Then assign it on the server in the background
      try {
        await api.assignTicket(ticket.ticket_id);
        // Fetch fresh data to sync with server
        await loadTickets();
        // Update selected ticket with fresh data
        setTickets(prevTickets => {
          const updatedTicket = prevTickets.find(t => t.ticket_id === ticket.ticket_id);
          if (updatedTicket) {
            setSelectedTicket(updatedTicket);
          }
          return prevTickets;
        });
      } catch (err) {
        console.error('Failed to assign ticket:', err);
        // Revert on error
        await loadTickets();
      }
    } else {
      setSelectedTicket(ticket);
    }
  }

  function handleLogout() {
    localStorage.removeItem('agent_token');
    localStorage.removeItem('agent_id');
    window.location.reload();
  }

  // Memoize filtered tickets to prevent unnecessary recalculations
  const { activeTickets, inProgressTickets, resolvedTickets, displayTickets } = useMemo(() => {
    const active = tickets.filter(t => t.status === 'active');
    const inProgress = tickets.filter(t => t.status === 'in_progress');
    const resolved = tickets.filter(t => t.status === 'resolved');
    
    const display = filter === 'active' ? active :
                    filter === 'in_progress' ? inProgress :
                    filter === 'resolved' ? resolved :
                    tickets;
    
    return { activeTickets: active, inProgressTickets: inProgress, resolvedTickets: resolved, displayTickets: display };
  }, [tickets, filter]);

  return (
    <div className='flex h-screen bg-gray-100'>
      {/* Sidebar - Ticket List */}
      <div className='w-80 bg-white border-r border-gray-200 flex flex-col'>
        <div className='p-4 border-b border-gray-200'>
          <div className='flex items-center justify-between mb-2'>
            <div>
          <h1 className='text-xl font-bold text-gray-800'>Agent Dashboard</h1>
          <p className='text-sm text-gray-500 mt-1'>Alkhidmat Support</p>
            </div>
            <div className='flex gap-3'>
              <button
                onClick={handleLogout}
                className='text-sm text-red-600 hover:text-red-800 font-medium'
              >
                Logout
              </button>
            </div>
          </div>
        </div>

        {/* Filter Tabs — Active (left) | Open Ticket | Resolved */}
        <div className='flex border-b border-gray-200'>
          <button
            onClick={() => setFilter('active')}
            className={`flex-1 px-4 py-2 text-sm font-medium ${
              filter === 'active' ? 'bg-yellow-50 text-yellow-700 border-b-2 border-yellow-600' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Active ({activeTickets.length})
          </button>
          <button
            onClick={() => setFilter('in_progress')}
            className={`flex-1 px-4 py-2 text-sm font-medium ${
              filter === 'in_progress' ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Open Ticket ({inProgressTickets.length})
          </button>
          <button
            onClick={() => setFilter('resolved')}
            className={`flex-1 px-4 py-2 text-sm font-medium ${
              filter === 'resolved' ? 'bg-green-50 text-green-600 border-b-2 border-green-600' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Resolved ({resolvedTickets.length})
          </button>
        </div>

        {/* Ticket List */}
        <div className='flex-1 overflow-y-auto'>
          {displayTickets.length === 0 ? (
            <div className='p-4 text-center text-gray-500 text-sm'>
              No {filter} tickets
            </div>
          ) : (
            displayTickets.map(ticket => (
              <div
                key={ticket.ticket_id}
                onClick={() => selectTicket(ticket)}
                className={`p-4 border-b border-gray-100 cursor-pointer hover:bg-gray-50 transition-colors ${
                  selectedTicket?.ticket_id === ticket.ticket_id ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''
                }`}
              >
                <div className='flex items-start justify-between'>
                  <div className='flex-1'>
                    <div className='flex items-center gap-2'>
                      <span className='text-sm font-semibold text-gray-800'>#{ticket.ticket_id.slice(0, 8)}</span>
                      <span className={`px-2 py-0.5 text-xs rounded-full ${
                        ticket.status === 'active' ? 'bg-yellow-100 text-yellow-800' :
                        ticket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {ticket.status === 'in_progress' ? 'Open' : ticket.status}
                      </span>
                    </div>
                    <p className='text-xs text-gray-500 mt-1'>
                      {new Date(ticket.created_at).toLocaleString()}
                    </p>
                    {ticket.phone_number && (
                      <p className='text-xs text-gray-500 mt-0.5 flex items-center gap-1'>
                        <span>📞</span> {ticket.phone_number}
                      </p>
                    )}
                    {ticket.response?.content && (
                      <p className='text-sm text-gray-700 mt-2 line-clamp-2'>
                        {ticket.response.content}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className='flex-1 flex flex-col'>
        {selectedTicket ? (
          <>
            {/* Chat Header */}
            <div className='bg-white border-b border-gray-200 p-4 flex items-center justify-between'>
              <div>
                <h2 className='text-lg font-semibold text-gray-800'>Ticket #{selectedTicket.ticket_id.slice(0, 8)}</h2>
                <div className='flex flex-wrap items-center gap-x-3 gap-y-0.5 mt-0.5'>
                  <p className='text-sm text-gray-500'>
                    {selectedTicket.session?.user_id ? `User: ${selectedTicket.session.user_id.slice(0, 8)}…` : 'Session Info'}
                  </p>
                  {selectedTicket.phone_number && (
                    <p className='text-sm text-gray-600 font-medium flex items-center gap-1'>
                      <span>📞</span> {selectedTicket.phone_number}
                    </p>
                  )}
                </div>
              </div>
              <div className='flex items-center gap-2'>
                <span className={`px-3 py-1 text-xs rounded-full ${
                  selectedTicket.status === 'active' ? 'bg-yellow-100 text-yellow-800' :
                  selectedTicket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {selectedTicket.status === 'in_progress' ? 'Open' : selectedTicket.status}
                </span>
                {selectedTicket.status !== 'resolved' && (
                <button
                    onClick={() => resolveTicket(selectedTicket.ticket_id)}
                  className='px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium'
                >
                  Resolve
                </button>
                )}
              </div>
            </div>

            {/* Chat Messages */}
            <div 
              ref={chatContainerRef}
              className='flex-1 overflow-y-auto bg-[#e5ddd5] p-4'
            >
              <div className='max-w-4xl mx-auto space-y-3'>
                {loadingChat ? (
                  <div className='flex items-center justify-center py-12'>
                    <Spinner />
                    <span className='ml-3 text-gray-600'>Loading messages...</span>
                  </div>
                ) : chatMessages.length === 0 ? (
                  <div className='text-center text-gray-500 py-8'>
                    No messages yet. Start the conversation!
                  </div>
                ) : (
                  <>
                    {chatMessages.map((msg, idx) => {
                      // Determine message sender: user, assistant (RAG), or agent (human)
                      const isUser = msg.sender === 'user' || msg.role === 'user' || msg.type === 'query';
                      const isAgentMessage = msg.sender === 'agent'; // Human agent messages
                      const isAssistant = msg.sender === 'assistant' || msg.role === 'assistant' || (msg.type === 'response' && !isAgentMessage);
                      
                      // Display: user messages on left (white), assistant/agent messages on right (green)
                      const isRightAligned = isAssistant || isAgentMessage;
                      
                    return (
                      <div key={idx} className={`flex ${isRightAligned ? 'justify-end' : 'justify-start'} mb-2`}>
                        <div
                          className={`max-w-[75%] px-4 py-2 rounded-lg shadow-sm ${
                            isRightAligned ? 'bg-[#dcf8c6]' : 'bg-white'
                          }`}
                          dir='auto'
                        >
                          {isAssistant && !isAgentMessage && (
                            <span className='text-xs text-gray-500 italic block mb-1'>AI Assistant</span>
                          )}
                          {isAgentMessage && (
                            <span className='text-xs text-gray-500 italic block mb-1'>You</span>
                          )}
                          {(msg.image_url || msg.image_data_url) && (
                            <div className='mb-2'>
                              <img
                                src={msg.image_url || msg.image_data_url}
                                alt="User uploaded image"
                                className='max-h-64 max-w-full rounded-lg object-contain'
                              />
                            </div>
                          )}
                          <p className='text-sm text-gray-800 whitespace-pre-wrap'>
                            {(msg.image_url || msg.image_data_url)
                              ? (msg.content || '').split('\n\nText detected in attached image:')[0].trim()
                              : msg.content}
                          </p>
                          <p className='text-xs text-gray-500 mt-1'>
                              {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}
                          </p>
                        </div>
                      </div>
                    );
                    })}
                    {/* Invisible div at the bottom for scrolling */}
                    <div ref={chatMessagesEndRef} />
                  </>
                )}
              </div>
            </div>

            {/* Chat Input */}
            {selectedTicket.status !== 'resolved' && (
            <div className='bg-white border-t border-gray-200 p-4'>
              <div className='max-w-4xl mx-auto'>
                <div className='flex items-end gap-2'>
                  <textarea
                    value={newMessage}
                    onChange={e => setNewMessage(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    placeholder='Type your message...'
                    className='flex-1 px-4 py-2 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500'
                    rows={1}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!newMessage.trim()}
                    className='px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium'
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
            )}
          </>
        ) : (
          <div className='flex-1 flex items-center justify-center bg-gray-50'>
            <div className='text-center'>
              <p className='text-gray-500 text-lg'>Select a ticket to start chatting</p>
              <p className='text-gray-400 text-sm mt-2'>Choose a ticket from the sidebar</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AgentDashboard;
