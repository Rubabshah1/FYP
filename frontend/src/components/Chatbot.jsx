import { useState, useEffect } from 'react';
import { useImmer } from 'use-immer';
import api from '@/api';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';

function Chatbot() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  const [isAgentChat, setIsAgentChat] = useState(false);
  const [ticketId, setTicketId] = useState(null);

  const isLoading = messages.length && messages[messages.length - 1].loading;

  useEffect(() => {
    // Get session from localStorage (user must be logged in first)
    const storedSessionId = localStorage.getItem('user_session_id');
    if (storedSessionId) {
      // Validate session format - should be a UUID (contains hyphens and is 36 chars)
      // Old tokens are longer and don't have the UUID format
      const isUUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(storedSessionId);
      if (isUUID) {
        setSessionId(storedSessionId);
      } else {
        // Old token format - clear it
        console.log('Old session token detected, clearing...');
        localStorage.removeItem('user_session_id');
        localStorage.removeItem('user_id');
        // Redirect to login
        window.location.hash = '#/user';
        window.location.reload();
      }
    }
    // Don't create session automatically - user must login first via OTP
  }, []);

  async function submitNewMessage() {
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || isLoading || !sessionId) return;

    setMessages(draft => [...draft,
      { role: 'user', content: trimmedMessage },
      { role: 'assistant', content: '', sources: [], loading: true }
    ]);
    setNewMessage('');

    try {
      const { answer = '', sources = [], agent_chat = false } = await api.sendChatMessage(sessionId, trimmedMessage);
      
      if (agent_chat) {
        setIsAgentChat(true);
      }
      
      setMessages(draft => {
        draft[draft.length - 1].content = answer;
        draft[draft.length - 1].sources = sources;
        draft[draft.length - 1].loading = false;
      });
    } catch (err) {
      console.log('Chat error:', err);
      
      // Handle 401 Unauthorized - session expired or invalid
      if (err.status === 401) {
        // Clear old session
        localStorage.removeItem('user_session_id');
        localStorage.removeItem('user_id');
        setSessionId(null);
        
        // Show error message
        setMessages(draft => {
          draft[draft.length - 1].loading = false;
          draft[draft.length - 1].error = true;
          draft[draft.length - 1].content = err.data?.detail || 'Session expired. Please login again.';
        });
        
        // Optionally redirect to login page after a delay
        setTimeout(() => {
          window.location.hash = '#/user';
          window.location.reload();
        }, 2000);
      } else {
        setMessages(draft => {
          draft[draft.length - 1].loading = false;
          draft[draft.length - 1].error = true;
        });
      }
    }
  }

  async function requestHumanAgent() {
    if (!sessionId) {
      const { session_id } = await api.createChat();
      setSessionId(session_id);
      localStorage.setItem('user_session_id', session_id);
    }
    
    try {
      const { ticket_id } = await api.createTicket(sessionId || localStorage.getItem('user_session_id'), 'User requested to chat with human agent');
      setTicketId(ticket_id);
      setIsAgentChat(true);
      
      setMessages(draft => [...draft,
        { role: 'system', content: 'Your request has been sent to our support team. An agent will be with you shortly.', timestamp: new Date().toISOString() }
      ]);
    } catch (err) {
      console.error('Failed to create ticket:', err);
    }
  }

  return (
    <div className='relative grow flex flex-col gap-6 pt-6'>
      {messages.length === 0 && (
        <div className='mt-3 font-urbanist text-primary-white text-xl font-light space-y-2'>
          <p>👋 Welcome!</p>
          <p>I am Alkhidmat AI Chatbot, your personal assistant. How can I help you today?</p>
          <button
            onClick={requestHumanAgent}
            className='mt-4 px-4 py-2 bg-[#25D366] text-white rounded-lg hover:bg-[#20BA5A] transition-colors text-sm font-medium'
          >
            💬 Chat with Human Agent
          </button>
        </div>
      )}
      {!isAgentChat && messages.length > 0 && (
        <div className='flex justify-center'>
          <button
            onClick={requestHumanAgent}
            className='px-4 py-2 bg-[#25D366] text-white rounded-lg hover:bg-[#20BA5A] transition-colors text-sm font-medium'
          >
            💬 Need Human Help? Chat with Agent
          </button>
        </div>
      )}
      {isAgentChat && (
        <div className='bg-blue-100 text-blue-800 px-4 py-2 rounded-lg text-sm text-center'>
          You are now chatting with a human agent. Please wait for their response.
        </div>
      )}
      <ChatMessages
        messages={messages}
        isLoading={isLoading}
      />
      <ChatInput
        newMessage={newMessage}
        isLoading={isLoading}
        setNewMessage={setNewMessage}
        submitNewMessage={submitNewMessage}
      />
    </div>
  );
}

export default Chatbot;
