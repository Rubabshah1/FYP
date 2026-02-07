import { memo, useMemo } from 'react';
import Markdown from 'react-markdown';
import useAutoScroll from '@/hooks/useAutoScroll';
import Spinner from '@/components/Spinner';
import logo from '@/assets/images/alkhidmat.png';

// WhatsApp background pattern - darker green with subtle icons
const whatsappPattern = {
  backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='whatsapp-pattern' x='0' y='0' width='100' height='100' patternUnits='userSpaceOnUse'%3E%3Ccircle cx='20' cy='20' r='1' fill='%23000000' opacity='0.03'/%3E%3Ccircle cx='50' cy='30' r='1' fill='%23000000' opacity='0.03'/%3E%3Ccircle cx='80' cy='50' r='1' fill='%23000000' opacity='0.03'/%3E%3Ccircle cx='30' cy='70' r='1' fill='%23000000' opacity='0.03'/%3E%3Ccircle cx='70' cy='80' r='1' fill='%23000000' opacity='0.03'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100' height='100' fill='url(%23whatsapp-pattern)'/%3E%3C/svg%3E")`,
  backgroundColor: '#e5ddd5' // WhatsApp beige background
};

// Format timestamp like WhatsApp (e.g., "5:52 PM")
function formatTime(timestamp) {
  if (!timestamp) return '';
  try {
    const date = new Date(timestamp);
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, '0');
    return `${displayHours}:${displayMinutes} ${ampm}`;
  } catch (e) {
    return '';
  }
}

// Memoized message item component - WhatsApp style
const MessageItem = memo(({ message }) => {
  const { role, content, loading, error, timestamp, sources, sender } = message;
  // Support both role and sender fields, and handle agent messages
  const actualRole = role || sender || 'assistant';
  const isUser = actualRole === 'user';
  const isAgent = actualRole === 'agent' || sender === 'agent';
  
  // Ensure content is always a string
  const messageContent = content || '';
  
  // Debug: Log message details if there's an issue
  if (loading && !isUser) {
    console.log('[ChatMessages] Loading message:', { role, actualRole, isUser, loading });
  }
  
  // WhatsApp colors - exact match
  const bubbleColor = isUser 
    ? 'bg-[#dcf8c6]' // Light green for sent messages (WhatsApp exact color)
    : 'bg-white';     // White for received messages
  
  const textColor = isUser ? 'text-[#111b21]' : 'text-[#111b21]';
  const alignment = isUser ? 'justify-end' : 'justify-start';
  
  return (
    <div className={`flex ${alignment} mb-1 px-2`}>
      {!isUser && (
        <div className="flex-shrink-0 mr-1 self-end mb-0.5">
          <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center overflow-hidden ring-1 ring-white/30">
            <img src={logo} alt="Alkhidmat" className="w-full h-full object-cover" />
          </div>
        </div>
      )}
      
      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[75%] sm:max-w-[65%] md:max-w-[55%]`}>
        <div
          className={`px-3 py-2 rounded-lg ${bubbleColor} ${textColor} ${
            isUser 
              ? 'rounded-tr-none' // Sent messages: rounded except top-right
              : 'rounded-tl-none' // Received messages: rounded except top-left
          }`}
          style={{
            wordWrap: 'break-word',
            wordBreak: 'break-word',
            boxShadow: '0 1px 0.5px rgba(0,0,0,0.13)',
            position: 'relative',
            maxWidth: '100%'
          }}
        >
          {loading && !isUser ? (
            <div className="flex items-center gap-1.5 py-0.5">
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          ) : (actualRole === 'assistant' || actualRole === 'agent') ? (
            <div className="text-sm leading-relaxed">
              {messageContent ? (
                <Markdown 
                  components={{
                    p: ({node, ...props}) => <p className="mb-1 last:mb-0" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc ml-4 mb-1" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal ml-4 mb-1" {...props} />,
                    li: ({node, ...props}) => <li className="mb-0.5" {...props} />,
                  }}
                >
                  {messageContent}
                </Markdown>
              ) : (
                <span className="text-gray-400 italic">No response received</span>
              )}
            </div>
          ) : (
            <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">{messageContent}</div>
          )}
          
          {error && (
            <div className='flex items-center gap-1 text-xs text-red-600 mt-1'>
              <svg className='h-3.5 w-3.5' fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>Error</span>
            </div>
          )}
        </div>
        
        {/* Timestamp and read receipt - WhatsApp style */}
        <div className={`flex items-center gap-1 mt-0.5 px-1 ${isUser ? 'flex-row-reverse' : ''}`}>
          <span className="text-[11px] text-gray-500 select-none whitespace-nowrap">
            {timestamp ? formatTime(timestamp) : formatTime(new Date().toISOString())}
          </span>
          {isUser && (
            <span className="text-[11px] text-gray-500 flex items-center ml-0.5">
              {/* Double checkmark for sent messages - WhatsApp style */}
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16">
                <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576 6.636 10.07Zm-1.138-1.138L1.47 2.294l4.338 2.761L5.498 8.932Z"/>
              </svg>
            </span>
          )}
        </div>
      </div>
      
      {isUser && (
        <div className="flex-shrink-0 ml-1 self-end mb-0.5">
          <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center overflow-hidden ring-1 ring-white/30">
            <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
});

MessageItem.displayName = 'MessageItem';

function ChatMessages({ messages, isLoading }) {
  const scrollContentRef = useAutoScroll(messages, isLoading);
  
  // Memoize messages list to prevent unnecessary re-renders
  const memoizedMessages = useMemo(() => {
    return messages.map((msg, idx) => ({
      ...msg,
      id: msg.id || `msg-${idx}-${msg.role}-${msg.content?.slice(0, 20)}`,
      timestamp: msg.timestamp || new Date().toISOString()
    }));
  }, [messages]);
  
  return (
    <div
      ref={scrollContentRef}
      className='flex-1 overflow-y-auto py-2 min-h-0'
      style={whatsappPattern}
    >
      {memoizedMessages.length === 0 ? (
        <div className="flex items-center justify-center h-full min-h-[400px]">
          <div className="text-center text-gray-700">
            <div className="mb-4">
              <svg className="w-16 h-16 mx-auto opacity-50" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
              </svg>
            </div>
            <p className="text-lg mb-2 font-medium">👋 Welcome!</p>
            <p className="text-sm opacity-70">Start a conversation with Alkhidmat AI</p>
          </div>
        </div>
      ) : (
        <div className="space-y-0.5">
          {memoizedMessages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
        </div>
      )}
    </div>
  );
}

export default memo(ChatMessages);
