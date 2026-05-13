import { useState, useEffect, useRef, useCallback } from 'react';
import { useImmer } from 'use-immer';
import api from '@/api';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';
import logo from '@/assets/images/alkhidmat.png';
import { useWebSocket } from '@/hooks/useWebSocket';

function Chatbot() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  const [isAgentChat, setIsAgentChat] = useState(false);
  const [ticketId, setTicketId] = useState(null);
  const lastMessageCountRef = useRef(0);

  // Name collection state
  const [userName, setUserName] = useState(() => localStorage.getItem('user_name') || '');
  const [namePromptVisible, setNamePromptVisible] = useState(false);
  const [nameInput, setNameInput] = useState('');
  const [nameSubmitting, setNameSubmitting] = useState(false);

  const isLoading = messages.length && messages[messages.length - 1].loading;

  useEffect(() => {
    // Get session from localStorage (user must be logged in first)
    const storedSessionId = localStorage.getItem('user_session_id');
    if (storedSessionId) {
      const isUUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(storedSessionId);
      if (isUUID) {
        setSessionId(storedSessionId);
        loadChatHistory(storedSessionId);
      } else {
        localStorage.removeItem('user_session_id');
        localStorage.removeItem('user_id');
        window.location.hash = '#/user';
        window.location.reload();
      }
    }
  }, []);

  // WebSocket connection for real-time agent messages
  const handleWebSocketMessage = useCallback((data) => {
    console.log('[Chatbot] WebSocket message received:', data);

    if (data.type === 'new_message' && data.message) {
      // Add new agent message to chat
      setMessages(draft => {
        const newMsg = {
          id: `agent-${Date.now()}-${Math.random()}`,
          role: 'agent',
          sender: 'agent',
          content: data.message.content,
          timestamp: data.message.timestamp || new Date().toISOString(),
          sources: [],
          ...(data.message.response_id && { response_id: data.message.response_id })
        };
        draft.push(newMsg);
        lastMessageCountRef.current = draft.length;
      });
    }

    // ── Route back to AI when agent resolves the ticket ──
    if (data.type === 'ticket_resolved') {
      setIsAgentChat(false);
      setTicketId(null);
      setMessages(draft => {
        draft.push({
          id: `system-${Date.now()}`,
          role: 'system',
          sender: 'system',
          content: '✅ Your conversation with the agent has ended. You are now back with the AI assistant — feel free to ask anything!',
          timestamp: new Date().toISOString(),
          sources: []
        });
        lastMessageCountRef.current = draft.length;
      });
    }
  }, [setMessages]);

  const handleWebSocketError = useCallback((error) => {
    console.error('[Chatbot] WebSocket error:', error);
  }, []);

  const handleWebSocketOpen = useCallback(() => {
    console.log('[Chatbot] WebSocket connected');
  }, []);

  const handleWebSocketClose = useCallback((event) => {
    console.log('[Chatbot] WebSocket closed:', event.code, event.reason);
  }, []);

  const { connected: wsConnected } = useWebSocket(
    sessionId ? `/ws/user/${sessionId}` : null,
    {
      enabled: !!sessionId,          // keep WS alive always so we receive ticket_resolved too
      onMessage: handleWebSocketMessage,
      onError: handleWebSocketError,
      onOpen: handleWebSocketOpen,
      onClose: handleWebSocketClose
    }
  );

  async function loadChatHistory(sid, checkForNewMessages = false) {
    try {
      const chatData = await api.createChat();
      if (chatData.chat_history && chatData.chat_history.length > 0) {
        const historyMessages = chatData.chat_history.map(msg => ({
          role: msg.role || msg.sender || 'assistant',
          content: (msg.image_url && (msg.role === 'user' || msg.sender === 'user'))
            ? (msg.content || '').split('\n\nText detected in attached image:')[0].trim()
            : msg.content,
          timestamp: msg.timestamp,
          sources: (msg.role === 'assistant' || msg.sender === 'assistant') ? [] : undefined,
          ...(msg.confidence && { confidence: msg.confidence }),
          ...(msg.domain && { domain: msg.domain }),
          ...(msg.sender && { sender: msg.sender }),
          ...(msg.image_url && { image_url: msg.image_url }),
        }));
        
        if (checkForNewMessages && historyMessages.length > lastMessageCountRef.current) {
          lastMessageCountRef.current = historyMessages.length;
        } else if (!checkForNewMessages) {
          lastMessageCountRef.current = historyMessages.length;
        }

        setMessages(historyMessages);

        // Sync name from backend if available
        if (chatData.user_name) {
          localStorage.setItem('user_name', chatData.user_name);
          setUserName(chatData.user_name);
          setNamePromptVisible(false);
        }

        // ── Sync agent chat state from backend (Active tickets only) ──
        if (chatData.is_agent_chat) {
          setIsAgentChat(true);
          if (chatData.active_ticket_id) setTicketId(chatData.active_ticket_id);
          console.log('[Chatbot] Active agent ticket found:', chatData.active_ticket_id);
        } else {
          setIsAgentChat(false);
          setTicketId(null);
        }
      } else {
        if (!checkForNewMessages) {
          // Even if history is empty, check if backend returned a name
          const finalName = chatData.user_name || localStorage.getItem('user_name');

          if (finalName) {
            if (chatData.user_name) {
              localStorage.setItem('user_name', chatData.user_name);
              setUserName(chatData.user_name);
            }
            // Returning user with known name but no history
            const welcomeMessage = {
              role: 'assistant',
              content: `👋 Welcome back, **${finalName}**!\n\nHow can I help you today?`,
              timestamp: new Date().toISOString(),
              sources: []
            };
            setMessages([welcomeMessage]);
            lastMessageCountRef.current = 1;
          } else {
            // Brand-new user — show name prompt first
            setNamePromptVisible(true);
            const welcomeMessage = {
              role: 'assistant',
              content: '👋 Welcome to **Alkhidmat Foundation**!\n\nBefore we begin, what\'s your name?',
              timestamp: new Date().toISOString(),
              sources: []
            };
            setMessages([welcomeMessage]);
            lastMessageCountRef.current = 1;
          }
        }
      }
    } catch (err) {
      console.error('Failed to load chat history:', err);
      if (sid) {
        try {
          const historyData = await api.getChatHistory(sid);
          if (historyData.messages && historyData.messages.length > 0) {
            const historyMessages = historyData.messages.map(msg => ({
              role: msg.role || msg.sender || 'assistant',
              content: (msg.image_url && (msg.role === 'user' || msg.sender === 'user'))
                ? (msg.content || '').split('\n\nText detected in attached image:')[0].trim()
                : msg.content,
              timestamp: msg.timestamp,
              sources: (msg.role === 'assistant' || msg.sender === 'assistant') ? [] : undefined,
              ...(msg.confidence && { confidence: msg.confidence }),
              ...(msg.domain && { domain: msg.domain }),
              ...(msg.sender && { sender: msg.sender }),
              ...(msg.image_url && { image_url: msg.image_url }),
            }));
            setMessages(historyMessages);
            lastMessageCountRef.current = historyMessages.length;
          } else if (!checkForNewMessages) {
            setNamePromptVisible(true);
            const welcomeMessage = {
              role: 'assistant',
              content: '👋 Welcome to **Alkhidmat Foundation**!\n\nBefore we begin, what\'s your name?',
              timestamp: new Date().toISOString(),
              sources: []
            };
            setMessages([welcomeMessage]);
            lastMessageCountRef.current = 1;
          }
        } catch (err2) {
          console.error('Failed to load session chat history:', err2);
          if (!checkForNewMessages) {
            setNamePromptVisible(true);
            const welcomeMessage = {
              role: 'assistant',
              content: '👋 Welcome to **Alkhidmat Foundation**!\n\nBefore we begin, what\'s your name?',
              timestamp: new Date().toISOString(),
              sources: []
            };
            setMessages([welcomeMessage]);
            lastMessageCountRef.current = 1;
          }
        }
      }
    }
  }

  // ── Handle name submission ──────────────────────────────────────────────────
  function handleNameSubmit(e) {
    e.preventDefault();
    const trimmed = nameInput.trim();
    if (!trimmed) return;
    setNameSubmitting(true);
    localStorage.setItem('user_name', trimmed);
    setUserName(trimmed);
    setNamePromptVisible(false);
    setNameInput('');

    // Optionally persist to backend
    api.updateUserName(trimmed).catch(err => console.warn('[Name] Could not persist name to backend:', err));

    // Add a greeting reply from the bot
    setMessages(draft => {
      draft.push({
        id: `assistant-name-${Date.now()}`,
        role: 'assistant',
        content: `Nice to meet you, **${trimmed}**! 😊 How can I help you today?`,
        timestamp: new Date().toISOString(),
        sources: []
      });
      lastMessageCountRef.current = draft.length;
    });
    setNameSubmitting(false);
  }

  async function submitNewMessage(messageOrFormData) {
    // Block chat until name is collected
    if (namePromptVisible) return;

    const isFormData = messageOrFormData instanceof FormData;
    const imageFile = isFormData ? messageOrFormData.get('image') : null;
    const hasImage = Boolean(imageFile);
    const localImageUrl = hasImage ? URL.createObjectURL(imageFile) : null;

    let trimmedMessage;
    if (isFormData) {
      trimmedMessage = messageOrFormData.get('message')?.trim() || '';
    } else {
      trimmedMessage = messageOrFormData?.trim() || '';
    }

    if (!trimmedMessage && !isFormData) return;
    if (isFormData && !trimmedMessage && !hasImage) return;
    if (isLoading || !sessionId) return;

    if (!isFormData) setNewMessage('');

    const userMessageId = `user-${Date.now()}-${Math.random()}`;
    const assistantMessageId = `assistant-${Date.now()}-${Math.random()}`;
    const userContent = trimmedMessage || '';

    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: userContent,
      timestamp: new Date().toISOString(),
      ...(hasImage ? { imageUrl: localImageUrl, imageName: imageFile?.name } : {})
    };

    // ── "Okay, wait — let me check…" loading message ──────────────────────────
    // Skip the loading bubble when already in agent-chat mode — message is forwarded silently
    const loadingAssistantMessage = isAgentChat ? null : {
      id: assistantMessageId,
      role: 'assistant',
      content: 'Okay, wait — let me check… 🔍',
      sources: [],
      loading: true,
      timestamp: new Date().toISOString()
    };

    setMessages(draft => {
      draft.push(userMessage);
      if (loadingAssistantMessage) draft.push(loadingAssistantMessage);
    });

    try {
      let responseData;
      if (isFormData) {
        responseData = await api.sendChatMessageWithImage(sessionId, messageOrFormData);
      } else {
        responseData = await api.sendChatMessage(sessionId, trimmedMessage);
      }

      const { answer = '', sources = [], agent_chat = false, ticket_id, ocr_text } = responseData;

      if (agent_chat) {
        setIsAgentChat(true);
        if (ticket_id) setTicketId(ticket_id);
      }

      if (ocr_text && typeof ocr_text === 'string' && ocr_text.trim()) {
        setMessages(draft => {
          const userIndex = draft.findIndex(msg => msg.id === userMessageId);
          if (userIndex !== -1) draft[userIndex].ocrText = ocr_text;
        });
      }

      // ── If agent-chat and no answer: backend forwarded to agent, no AI reply ──
      // Remove the loading bubble (if any) and don't show a blank AI message.
      if (agent_chat && !answer) {
        setMessages(draft => {
          const idx = draft.findIndex(msg => msg.id === assistantMessageId);
          if (idx !== -1) draft.splice(idx, 1);
          lastMessageCountRef.current = draft.length;
        });
      } else {
        setMessages(draft => {
          const assistantIndex = draft.findIndex(msg => msg.id === assistantMessageId);
          if (assistantIndex !== -1) {
            draft[assistantIndex].content = answer;
            draft[assistantIndex].sources = sources || [];
            draft[assistantIndex].loading = false;
            draft[assistantIndex].error = false;
            draft[assistantIndex].timestamp = new Date().toISOString();
            draft[assistantIndex].role = 'assistant';
          }
        });
      }

      setMessages(current => {
        lastMessageCountRef.current = current.length;
        return current;
      });

    } catch (err) {
      console.error('❌ Chat error:', err);

      if (err.status === 401) {
        localStorage.removeItem('user_session_id');
        localStorage.removeItem('user_id');
        setSessionId(null);
        setMessages(draft => {
          const assistantIndex = draft.findIndex(msg => msg.id === assistantMessageId);
          if (assistantIndex !== -1) {
            draft[assistantIndex].loading = false;
            draft[assistantIndex].error = true;
            draft[assistantIndex].content = err.data?.detail || 'Session expired. Please login again.';
            draft[assistantIndex].role = 'assistant';
          }
        });
        setTimeout(() => {
          window.location.hash = '#/user';
          window.location.reload();
        }, 2000);
      } else {
        setMessages(draft => {
          const assistantIndex = draft.findIndex(msg => msg.id === assistantMessageId);
          if (assistantIndex !== -1) {
            draft[assistantIndex].loading = false;
            draft[assistantIndex].error = true;
            draft[assistantIndex].content = 'Failed to get response. Please try again.';
            draft[assistantIndex].role = 'assistant';
          }
        });
      }
    }
  }

  return (
    <div className='relative flex flex-col h-full bg-white'>
      {/* WhatsApp-style header */}
      <div className='flex-shrink-0 bg-[#075e54] text-white px-4 py-3 shadow-md z-10'>
        <div className='flex items-center justify-between'>
          <div className='flex items-center gap-3'>
            <button
              onClick={() => {
                localStorage.removeItem('user_session_id');
                localStorage.removeItem('user_id');
                window.location.hash = '';
                window.location.reload();
              }}
              className='p-1.5 hover:bg-white/10 rounded-full transition-colors -ml-1'
              title="Back"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center overflow-hidden ring-2 ring-white/30">
              <img src={logo} alt="Alkhidmat" className="w-full h-full object-cover" />
            </div>
            <div>
              <h2 className='font-semibold text-base'>Alkhidmat AI</h2>
              <p className='text-xs text-white/90 flex items-center gap-1'>
                {userName ? (
                  <>
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                    Always online
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                    Always online
                  </>
                )}
              </p>
            </div>
          </div>
          <div className='flex items-center gap-2'>
            <button className='p-2 hover:bg-white/10 rounded-full transition-colors' title="Search">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
            <button
              className='p-2 hover:bg-white/10 rounded-full transition-colors text-white'
              title="Logout"
              onClick={() => {
                localStorage.removeItem('user_session_id');
                localStorage.removeItem('user_id');
                localStorage.removeItem('user_name');
                window.location.hash = '#/user';
                window.location.reload();
              }}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Chat messages area */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <ChatMessages
          messages={messages}
          isLoading={isLoading}
        />
      </div>

      {/* Agent chat status banner */}
      {isAgentChat && (
        <div className='flex-shrink-0 bg-blue-500 text-white px-4 py-2 text-sm text-center shadow-md z-10'>
          <div className='flex items-center justify-center gap-2'>
            <svg className="w-4 h-4 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
            </svg>
            You are now chatting with a human agent.
          </div>
        </div>
      )}

      {/* Name prompt overlay (shows instead of normal chat input) */}
      {namePromptVisible ? (
        <div className="flex-shrink-0 bg-white border-t border-gray-200 p-4">
          <form onSubmit={handleNameSubmit} className="flex items-center gap-2">
            <input
              type="text"
              value={nameInput}
              onChange={e => setNameInput(e.target.value)}
              placeholder="Enter your name…"
              autoFocus
              className="flex-1 px-4 py-2.5 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-[#075e54] text-sm"
            />
            <button
              type="submit"
              disabled={!nameInput.trim() || nameSubmitting}
              className="px-5 py-2.5 bg-[#075e54] text-white rounded-full text-sm font-medium hover:bg-[#064e45] disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {nameSubmitting ? '…' : 'Continue'}
            </button>
          </form>
        </div>
      ) : (
        /* Normal input area */
        <div className="flex-shrink-0">
          <ChatInput
            newMessage={newMessage}
            isLoading={isLoading}
            setNewMessage={setNewMessage}
            submitNewMessage={submitNewMessage}
          />
        </div>
      )}
    </div>
  );
}

export default Chatbot;