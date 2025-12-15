import { memo, useMemo } from 'react';
import Markdown from 'react-markdown';
import useAutoScroll from '@/hooks/useAutoScroll';
import Spinner from '@/components/Spinner';
import userIcon from '@/assets/images/user.svg';
import errorIcon from '@/assets/images/error.svg';

// Memoized message item component to prevent unnecessary re-renders
const MessageItem = memo(({ message }) => {
  const { role, content, loading, error } = message;
  const isUser = role === 'user';
  const bubbleColor = isUser ? 'bg-[#dcf8c6]' : 'bg-white';
  const alignment = isUser ? 'justify-end' : 'justify-start';
  
  return (
    <div className={`flex ${alignment}`}>
      {!isUser && (
        <img
          className='h-[24px] w-[24px] shrink-0 self-end mr-2'
          src={userIcon}
          alt='assistant'
          style={{ visibility: 'hidden' }}
        />
      )}
      <div
        className={`max-w-[90%] sm:max-w-[75%] px-3 py-2 rounded-2xl shadow-sm ${bubbleColor}`}
        dir='auto'
      >
        <div className='markdown-container text-main-text'>
          {loading ? (
            <Spinner />
          ) : role === 'assistant' ? (
            <Markdown>{content}</Markdown>
          ) : (
            <div className='whitespace-pre-line' dir='auto'>{content}</div>
          )}
        </div>
        {error && (
          <div className='flex items-center gap-1 text-sm text-error-red mt-2'>
            <img className='h-5 w-5' src={errorIcon} alt='error' />
            <span>Error generating the response</span>
          </div>
        )}
      </div>
      {isUser && (
        <img
          className='h-[24px] w-[24px] shrink-0 self-end ml-2'
          src={userIcon}
          alt='user'
        />
      )}
    </div>
  );
});

MessageItem.displayName = 'MessageItem';

function ChatMessages({ messages, isLoading }) {
  const scrollContentRef = useAutoScroll(isLoading);
  
  // Memoize messages list to prevent unnecessary re-renders
  const memoizedMessages = useMemo(() => {
    return messages.map((msg, idx) => ({
      ...msg,
      id: msg.id || `msg-${idx}-${msg.role}-${msg.content?.slice(0, 20)}`
    }));
  }, [messages]);
  
  return (
    <div
      ref={scrollContentRef}
      className='grow space-y-3 px-1 py-2 bg-[#e5ddd5] rounded-2xl'
    >
      {memoizedMessages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </div>
  );
}

export default memo(ChatMessages);