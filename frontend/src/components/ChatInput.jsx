import useAutosize from '@/hooks/useAutosize';
import sendIcon from '@/assets/images/send.svg';

function ChatInput({ newMessage, isLoading, setNewMessage, submitNewMessage }) {
  const textareaRef = useAutosize(newMessage);

  function handleKeyDown(e) {
    if(e.keyCode === 13 && !e.shiftKey && !isLoading) {
      e.preventDefault();
      submitNewMessage();
    }
  }
  
  return(
    <div className='sticky bottom-0 shrink-0 bg-[#e5ddd5] py-4'>
      <div className='p-1.5 bg-[#c2d5c8] rounded-3xl z-50 font-mono origin-bottom animate-chat duration-400 shadow'>
        <div className='pr-0.5 bg-white relative shrink-0 rounded-3xl overflow-hidden ring-0 focus-within:ring-2 focus-within:ring-primary-blue transition-all shadow'>
          <textarea
            className='block w-full max-h-[140px] py-2 px-4 pr-11 bg-white rounded-3xl resize-none placeholder:text-gray-500 placeholder:leading-4 placeholder:-translate-y-1 sm:placeholder:leading-normal sm:placeholder:translate-y-0 focus:outline-hidden'
            ref={textareaRef}
            rows='1'
            value={newMessage}
            onChange={e => setNewMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            dir='auto'
            lang='ur'
          />
          <button
            className='absolute top-1/2 -translate-y-1/2 right-3 p-1 rounded-md hover:bg-primary-blue/20'
            onClick={submitNewMessage}
          >
            <img src={sendIcon} alt='send' />
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInput;