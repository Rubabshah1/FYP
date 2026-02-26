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
        // Load chat history when session is found
        loadChatHistory(storedSessionId);
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
  
  // WebSocket connection for real-time agent messages
  const handleWebSocketMessage = useCallback((data) => {
    console.log('[Chatbot] WebSocket message received:', data);
    if (data.type === 'new_message' && data.message) {
      // Add new agent message to chat
      setMessages(draft => {
        const newMessage = {
          id: `agent-${Date.now()}-${Math.random()}`,
          role: 'agent',
          sender: 'agent',
          content: data.message.content,
          timestamp: data.message.timestamp || new Date().toISOString(),
          sources: [],
          ...(data.message.response_id && { response_id: data.message.response_id })
        };
        draft.push(newMessage);
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
      enabled: !!sessionId && isAgentChat,
      onMessage: handleWebSocketMessage,
      onError: handleWebSocketError,
      onOpen: handleWebSocketOpen,
      onClose: handleWebSocketClose
    }
  );

  async function loadChatHistory(sessionId, checkForNewMessages = false) {
    try {
      // Try to get chat history from createChat endpoint (includes all user history)
      const chatData = await api.createChat();
      if (chatData.chat_history && chatData.chat_history.length > 0) {
        // Convert chat history to message format
        const historyMessages = chatData.chat_history.map(msg => ({
          role: msg.role || msg.sender || 'assistant', // Support both role and sender fields
          content: msg.content,
          timestamp: msg.timestamp,
          sources: (msg.role === 'assistant' || msg.sender === 'assistant') ? [] : undefined,
          // Include additional metadata if available
          ...(msg.confidence && { confidence: msg.confidence }),
          ...(msg.domain && { domain: msg.domain }),
          ...(msg.sender && { sender: msg.sender })
        }));
        
        // Check if we have new messages (for polling)
        if (checkForNewMessages && historyMessages.length > lastMessageCountRef.current) {
          console.log(`[Chatbot] New messages detected: ${historyMessages.length} total (was ${lastMessageCountRef.current})`);
          lastMessageCountRef.current = historyMessages.length;
        } else if (!checkForNewMessages) {
          lastMessageCountRef.current = historyMessages.length;
        }
        
        setMessages(historyMessages);
        
        // Check if user is in agent chat (has agent messages)
        const hasAgentMessages = historyMessages.some(msg => 
          msg.role === 'agent' || msg.sender === 'agent'
        );
        if (hasAgentMessages && !isAgentChat) {
          setIsAgentChat(true);
        }
      } else {
        // New user - no chat history, show welcome message
        if (!checkForNewMessages) {
          const welcomeMessage = {
            role: 'assistant',
            content: '👋 Welcome!\n\nStart a conversation with Alkhidmat AI',
            timestamp: new Date().toISOString(),
            sources: []
          };
          setMessages([welcomeMessage]);
          lastMessageCountRef.current = 1;
        }
      }
    } catch (err) {
      console.error('Failed to load chat history:', err);
      // If it fails, try to get session-specific history
      if (sessionId) {
        try {
          const historyData = await api.getChatHistory(sessionId);
          if (historyData.messages && historyData.messages.length > 0) {
            const historyMessages = historyData.messages.map(msg => ({
              role: msg.role || msg.sender || 'assistant',
              content: msg.content,
              timestamp: msg.timestamp,
              sources: (msg.role === 'assistant' || msg.sender === 'assistant') ? [] : undefined,
              ...(msg.confidence && { confidence: msg.confidence }),
              ...(msg.domain && { domain: msg.domain }),
              ...(msg.sender && { sender: msg.sender })
            }));
            
            if (checkForNewMessages && historyMessages.length > lastMessageCountRef.current) {
              console.log(`[Chatbot] New messages detected: ${historyMessages.length} total (was ${lastMessageCountRef.current})`);
              lastMessageCountRef.current = historyMessages.length;
            } else if (!checkForNewMessages) {
              lastMessageCountRef.current = historyMessages.length;
            }
            
            setMessages(historyMessages);
            
            // Check if user is in agent chat
            const hasAgentMessages = historyMessages.some(msg => 
              msg.role === 'agent' || msg.sender === 'agent'
            );
            if (hasAgentMessages && !isAgentChat) {
              setIsAgentChat(true);
            }
          } else if (!checkForNewMessages) {
            // New user - show welcome message
            const welcomeMessage = {
              role: 'assistant',
              content: '👋 Welcome!\n\nStart a conversation with Alkhidmat AI',
              timestamp: new Date().toISOString(),
              sources: []
            };
            setMessages([welcomeMessage]);
            lastMessageCountRef.current = 1;
          }
        } catch (err2) {
          console.error('Failed to load session chat history:', err2);
          if (!checkForNewMessages) {
            // New user - show welcome message
            const welcomeMessage = {
              role: 'assistant',
              content: '👋 Welcome!\n\nStart a conversation with Alkhidmat AI',
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

  // async function submitNewMessage() {
  //   const trimmedMessage = newMessage.trim();
  //   if (!trimmedMessage || isLoading || !sessionId) return;

  //   // Clear input immediately for better UX
  //   setNewMessage('');
    
  //   // Create unique IDs for tracking messages
  //   const userMessageId = `user-${Date.now()}-${Math.random()}`;
  //   const assistantMessageId = `assistant-${Date.now()}-${Math.random()}`;
    
  //   // Add user message immediately (optimistic update)
  //   const userMessage = { 
  //     id: userMessageId,
  //     role: 'user', 
  //     content: trimmedMessage, 
  //     timestamp: new Date().toISOString() 
  //   };
    
  //   // Add loading assistant message immediately
  //   const loadingAssistantMessage = { 
  //     id: assistantMessageId,
  //     role: 'assistant', 
  //     content: '', 
  //     sources: [], 
  //     loading: true, 
  //     timestamp: new Date().toISOString() 
  //   };
    
  //   // Add both messages at once
  //   setMessages(draft => [...draft, userMessage, loadingAssistantMessage]);

  //   try {
  //     const { answer = '', sources = [], agent_chat = false, ticket_id } = await api.sendChatMessage(sessionId, trimmedMessage);
      
  //     if (agent_chat) {
  //       setIsAgentChat(true);
  //       if (ticket_id) {
  //         setTicketId(ticket_id);
  //       }
  //     }
      
  //     // Update the assistant message by ID (more reliable than index)
  //     setMessages(draft => {
  //       const assistantIndex = draft.findIndex(msg => msg.id === assistantMessageId);
  //       if (assistantIndex !== -1) {
  //         draft[assistantIndex].content = answer;
  //         draft[assistantIndex].sources = sources || [];
  //         draft[assistantIndex].loading = false;
  //         draft[assistantIndex].error = false;
  //         draft[assistantIndex].timestamp = new Date().toISOString();
  //         // Ensure role is correct
  //         draft[assistantIndex].role = 'assistant';
  //       }
  //     });
      
  //     // Update message count after sending
  //     setMessages(current => {
  //       lastMessageCountRef.current = current.length;
  //       return current;
  //     });
  //   } catch (err) {
  //     console.log('Chat error:', err);
      
  //     // Handle 401 Unauthorized - session expired or invalid
  //     if (err.status === 401) {
  //       // Clear old session
  //       localStorage.removeItem('user_session_id');
  //       localStorage.removeItem('user_id');
  //       setSessionId(null);
        
  //       // Show error message
  //       setMessages(draft => {
  //         const assistantIndex = draft.findIndex(msg => msg.id === assistantMessageId);
  //         if (assistantIndex !== -1) {
  //           draft[assistantIndex].loading = false;
  //           draft[assistantIndex].error = true;
  //           draft[assistantIndex].content = err.data?.detail || 'Session expired. Please login again.';
  //           draft[assistantIndex].role = 'assistant';
  //         }
  //       });
        
  //       // Optionally redirect to login page after a delay
  //       setTimeout(() => {
  //         window.location.hash = '#/user';
  //         window.location.reload();
  //       }, 2000);
  //     } else {
  //       setMessages(draft => {
  //         const assistantIndex = draft.findIndex(msg => msg.id === assistantMessageId);
  //         if (assistantIndex !== -1) {
  //           draft[assistantIndex].loading = false;
  //           draft[assistantIndex].error = true;
  //           draft[assistantIndex].content = 'Failed to get response. Please try again.';
  //           draft[assistantIndex].role = 'assistant';
  //         }
  //       });
  //     }
  //   }
  // }
// replace submit new message , new
async function submitNewMessage(messageOrFormData) {
  // Detect if it's FormData or plain string
  const isFormData = messageOrFormData instanceof FormData;
  const imageFile = isFormData ? messageOrFormData.get('image') : null;
  const hasImage = Boolean(imageFile);
  // Create a dedicated object URL for the chat message (so ChatInput can revoke its own preview safely)
  const localImageUrl = hasImage ? URL.createObjectURL(imageFile) : null;
  
  let trimmedMessage;
  if (isFormData) {
    trimmedMessage = messageOrFormData.get('message')?.trim() || '';
  } else {
    trimmedMessage = messageOrFormData?.trim() || '';
  }
  
  console.log('📤 submitNewMessage:', { isFormData, trimmedMessage, hasImage });
  
  // Validation
  if (!trimmedMessage && !isFormData) {
    console.log('❌ Empty message');
    return;
  }
  if (isFormData && !trimmedMessage && !hasImage) {
    console.log('❌ Empty FormData (no text, no image)');
    return;
  }
  if (isLoading || !sessionId) {
    console.log('❌ Cannot send:', { isLoading, sessionId });
    return;
  }

  // Clear input immediately for better UX (only for text-only)
  if (!isFormData) {
    setNewMessage('');
  }
  
  // Create unique IDs for tracking messages
  const userMessageId = `user-${Date.now()}-${Math.random()}`;
  const assistantMessageId = `assistant-${Date.now()}-${Math.random()}`;
  
  // Determine user message content
  const userContent = trimmedMessage || (hasImage ? '' : '');
  
  console.log('💬 User message:', userContent);
  
  // Add user message immediately (optimistic update)
  const userMessage = { 
    id: userMessageId,
    role: 'user', 
    content: userContent, 
    timestamp: new Date().toISOString(),
    ...(hasImage ? { imageUrl: localImageUrl, imageName: imageFile?.name } : {})
  };
  
  // Add loading assistant message immediately
  const loadingAssistantMessage = { 
    id: assistantMessageId,
    role: 'assistant', 
    content: '', 
    sources: [], 
    loading: true, 
    timestamp: new Date().toISOString() 
  };
  
  // Add both messages at once
  setMessages(draft => [...draft, userMessage, loadingAssistantMessage]);

  try {
    let responseData;
    
    // ✅ Use appropriate API function based on input type
    if (isFormData) {
      console.log('🖼️ Sending with image via sendChatMessageWithImage');
      responseData = await api.sendChatMessageWithImage(sessionId, messageOrFormData);
    } else {
      console.log('📝 Sending text-only via sendChatMessage');
      responseData = await api.sendChatMessage(sessionId, trimmedMessage);
    }
    
    console.log('✅ Response received:', responseData);
    
    const { answer = '', sources = [], agent_chat = false, ticket_id, ocr_text } = responseData;
    
    if (agent_chat) {
      setIsAgentChat(true);
      if (ticket_id) {
        setTicketId(ticket_id);
      }
    }

    // If OCR text was extracted, attach it to the user message so it shows in the UI (instead of terminal logs)
    if (ocr_text && typeof ocr_text === 'string' && ocr_text.trim()) {
      setMessages(draft => {
        const userIndex = draft.findIndex(msg => msg.id === userMessageId);
        if (userIndex !== -1) {
          draft[userIndex].ocrText = ocr_text;
        }
      });
    }
    
    // Update the assistant message by ID (more reliable than index)
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
    
    // Update message count after sending
    setMessages(current => {
      lastMessageCountRef.current = current.length;
      return current;
    });
    
  } catch (err) {
    console.error('❌ Chat error:', err);
    
    // Handle 401 Unauthorized - session expired or invalid
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
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                Always online
              </p>
            </div>
          </div>
          <div className='flex items-center gap-2'>
            <button className='p-2 hover:bg-white/10 rounded-full transition-colors' title="Search">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
            <button className='p-2 hover:bg-white/10 rounded-full transition-colors' title="Menu">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Chat messages area - full height with WhatsApp background */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <ChatMessages
          messages={messages}
          isLoading={isLoading}
        />
      </div>

      {/* Human agent button - WhatsApp style floating button */}
      {!isAgentChat && messages.length > 0 && (
        <div className='absolute bottom-20 left-0 right-0 z-10 flex justify-center px-4 py-2 pointer-events-none'>
          <button
            onClick={requestHumanAgent}
            className='px-4 py-2 bg-[#25D366] text-white rounded-full shadow-lg hover:bg-[#20BA5A] transition-colors text-sm font-medium flex items-center gap-2 pointer-events-auto'
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
            </svg>
            Need Human Help?
          </button>
        </div>
      )}

      {/* Agent chat status */}
      {isAgentChat && (
        <div className='absolute top-15 left-0 right-0 z-10 bg-blue-500 text-white px-4 py-2 text-sm text-center shadow-md'>
          <div className='flex items-center justify-center gap-2'>
            <svg className="w-4 h-4 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            You are now chatting with a human agent. 
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="flex-shrink-0">
        <ChatInput
          newMessage={newMessage}
          isLoading={isLoading}
          setNewMessage={setNewMessage}
          submitNewMessage={submitNewMessage}
        />
      </div>
    </div>
  );
}

export default Chatbot;