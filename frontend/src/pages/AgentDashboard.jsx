import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useImmer } from 'use-immer';
import api from '@/api';
import logo from '@/assets/images/alkhidmat.png';

function AgentDashboard() {
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [chatMessages, setChatMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  const [filter, setFilter] = useState('active'); // active, in_progress, resolved
  const loadingRef = useRef(false); // Prevent concurrent requests

  // Load tickets with request deduplication
  const loadTickets = useCallback(async () => {
    if (loadingRef.current) return; // Skip if already loading
    loadingRef.current = true;
    try {
      const { tickets: ticketsData } = await api.listTickets(filter === 'all' ? null : filter);
      setTickets(ticketsData || []);
    } catch (err) {
      console.error('Failed to load tickets:', err);
    } finally {
      loadingRef.current = false;
    }
  }, [filter]);

  // Load chat messages with request deduplication
  const loadChatMessages = useCallback(async () => {
    if (!selectedTicket || loadingRef.current) return;
    loadingRef.current = true;
    try {
      const chat = await api.getTicketChat(selectedTicket.ticket_id);
      setChatMessages(chat.messages || []);
    } catch (err) {
      console.error('Failed to load chat messages:', err);
    } finally {
      loadingRef.current = false;
    }
  }, [selectedTicket, setChatMessages]);

  // Load tickets
  useEffect(() => {
    loadTickets();
    const interval = setInterval(loadTickets, 8000); // Reduced to 8 seconds (was 5)
    return () => clearInterval(interval);
  }, [loadTickets]);

  // Load chat messages when ticket is selected
  useEffect(() => {
    if (selectedTicket) {
      loadChatMessages();
      const interval = setInterval(loadChatMessages, 5000); // Reduced to 5 seconds (was 2)
      return () => clearInterval(interval);
    }
  }, [selectedTicket, loadChatMessages]);


  async function assignTicket(ticketId) {
    try {
      await api.assignTicket(ticketId);
      await loadTickets();
      const updatedTicket = tickets.find(t => t.ticket_id === ticketId);
      if (updatedTicket) {
        setSelectedTicket(updatedTicket);
      }
    } catch (err) {
      console.error('Failed to assign ticket:', err);
    }
  }

  async function resolveTicket(ticketId) {
    try {
      await api.resolveTicket(ticketId);
      await loadTickets();
      if (selectedTicket?.ticket_id === ticketId) {
        setSelectedTicket(null);
        setChatMessages([]);
      }
    } catch (err) {
      console.error('Failed to resolve ticket:', err);
    }
  }

  async function sendMessage() {
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || !selectedTicket) return;

    setChatMessages(draft => [...draft, {
      sender: 'agent',
      content: trimmedMessage,
      timestamp: new Date().toISOString()
    }]);
    setNewMessage('');

    try {
      await api.sendAgentMessage(selectedTicket.ticket_id, trimmedMessage, 'agent');
      await loadChatMessages();
    } catch (err) {
      console.error('Failed to send message:', err);
    }
  }

  function selectTicket(ticket) {
    setSelectedTicket(ticket);
    if (ticket.status === 'active') {
      assignTicket(ticket.ticket_id);
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
                onClick={() => {
                  localStorage.removeItem('agent_token');
                  localStorage.removeItem('agent_id');
                  window.location.hash = '';
                  window.location.reload();
                }}
                className='text-sm text-gray-600 hover:text-gray-800 font-medium'
              >
                ← Back to Welcome
              </button>
              <button
                onClick={handleLogout}
                className='text-sm text-red-600 hover:text-red-800 font-medium'
              >
                Logout
              </button>
            </div>
          </div>
        </div>

        {/* Filter Tabs */}
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
            In Progress ({inProgressTickets.length})
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
                        {ticket.status}
                      </span>
                    </div>
                    <p className='text-xs text-gray-500 mt-1'>
                      {new Date(ticket.created_at).toLocaleString()}
                    </p>
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
                <p className='text-sm text-gray-500'>
                  {selectedTicket.session?.user_id ? `User ID: ${selectedTicket.session.user_id}` : 'Session Info'}
                </p>
              </div>
              <div className='flex items-center gap-2'>
                <span className={`px-3 py-1 text-xs rounded-full ${
                  selectedTicket.status === 'active' ? 'bg-yellow-100 text-yellow-800' :
                  selectedTicket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {selectedTicket.status}
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
            <div className='flex-1 overflow-y-auto bg-[#e5ddd5] p-4'>
              <div className='max-w-4xl mx-auto space-y-3'>
                {chatMessages.length === 0 ? (
                  <div className='text-center text-gray-500 py-8'>
                    No messages yet. Start the conversation!
                  </div>
                ) : (
                  chatMessages.map((msg, idx) => {
                    const isAgent = msg.sender === 'agent' || msg.type === 'response';
                  return (
                    <div key={idx} className={`flex ${isAgent ? 'justify-end' : 'justify-start'}`}>
                      <div
                        className={`max-w-[75%] px-4 py-2 rounded-2xl shadow-sm ${
                          isAgent ? 'bg-[#dcf8c6]' : 'bg-white'
                        }`}
                        dir='auto'
                      >
                        <p className='text-sm text-gray-800 whitespace-pre-wrap'>{msg.content}</p>
                        <p className='text-xs text-gray-500 mt-1'>
                            {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}
                        </p>
                      </div>
                    </div>
                  );
                  })
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
