import useAutosize from '@/hooks/useAutosize';

function ChatInput({ newMessage, isLoading, setNewMessage, submitNewMessage }) {
  const textareaRef = useAutosize(newMessage);

  function handleKeyDown(e) {
    if(e.keyCode === 13 && !e.shiftKey && !isLoading) {
      e.preventDefault();
      submitNewMessage();
    }
  }
  
  return(
    <div className='sticky bottom-0 shrink-0 bg-[#f0f2f5] px-2 py-2 border-t border-gray-200'>
      <div className='flex items-end gap-1 bg-white rounded-3xl px-2 py-1.5 shadow-sm'>
        {/* Plus button (for attachments) - WhatsApp style */}
        <button
          className='flex-shrink-0 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors'
          title="Attach"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
        
        {/* Text input - WhatsApp style */}
        <textarea
          className='flex-1 max-h-[120px] py-2 px-3 bg-transparent resize-none placeholder:text-gray-500 focus:outline-none text-sm leading-5'
          ref={textareaRef}
          rows='1'
          value={newMessage}
          onChange={e => setNewMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message"
          dir='auto'
        />
        
        {/* Send button or microphone - WhatsApp style */}
        {newMessage.trim() ? (
          <button
            className='flex-shrink-0 p-2 text-[#25D366] hover:bg-gray-100 rounded-full transition-colors'
            onClick={submitNewMessage}
            disabled={isLoading}
            title="Send"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        ) : (
          <button
            className='flex-shrink-0 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors'
            title="Record voice message"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

export default ChatInput;
