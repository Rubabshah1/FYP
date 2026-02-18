
import { useRef, useState } from "react"; // ← Add useState
import useAutosize from "@/hooks/useAutosize";

function ChatInput({ newMessage, isLoading, setNewMessage, submitNewMessage }) {
  const textareaRef = useAutosize(newMessage);
  const fileInputRef = useRef(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  
  function handleKeyDown(e) {
    if (e.keyCode === 13 && !e.shiftKey && !isLoading) {
      e.preventDefault();
      // submitNewMessage();
      handleSubmit();
    }
  }
  //add new img function 
  const handleImageSelect = (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  
  // Validate file type
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file');
    return;
  }
  
  // Validate file size (max 5MB)
  if (file.size > 5 * 1024 * 1024) {
    alert('Image must be less than 5MB');
    return;
  }
  
  console.log('✅ Image selected:', file.name, file.size, 'bytes');
  setSelectedImage(file);
  setImagePreview(URL.createObjectURL(file));
};
// ADD THIS NEW FUNCTION: image submission
async function handleSubmit() {
  if (!newMessage.trim() && !selectedImage) {
    console.log('❌ Nothing to send');
    return;
  }
  if (isLoading) {
    console.log('❌ Already sending');
    return;
  }
  
  console.log('📦 Creating FormData...');
  const formData = new FormData();
  formData.append('message', newMessage || '');
  
  if (selectedImage) {
    formData.append('image', selectedImage);
    console.log('✅ FormData with image:', selectedImage.name);
  } else {
    console.log('✅ FormData text-only');
  }
  
  // Call parent function
  await submitNewMessage(formData);
  
  // Clear inputs
  setNewMessage('');
  if (imagePreview) {
    URL.revokeObjectURL(imagePreview);
    setSelectedImage(null);
    setImagePreview(null);
    fileInputRef.current.value = '';
  }
  console.log('✅ Inputs cleared');
}
  return (
    <div className="sticky bottom-0 shrink-0 bg-[#f0f2f5] px-2 py-2 border-t border-gray-200">
      {/* ✅ ADD THIS IMAGE PREVIEW */}
    {imagePreview && (
      <div className='mb-2 px-2'>
        <div className='relative inline-block bg-white rounded-lg p-2 shadow-sm'>
          <img
            src={imagePreview}
            alt="Preview"
            className='max-h-32 max-w-[200px] rounded'
          />
          <button
            onClick={() => {
              URL.revokeObjectURL(imagePreview);
              setSelectedImage(null);
              setImagePreview(null);
              fileInputRef.current.value = '';
              console.log('✅ Image removed');
            }}
            className='absolute -top-2 -right-2 bg-gray-700 text-white rounded-full p-1 hover:bg-gray-800 transition-colors'
            title="Remove image"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    )}
      <div className="flex items-end gap-1 bg-white rounded-3xl px-2 py-1.5 shadow-sm">
        {/* ✅ ADD THIS HIDDEN FILE INPUT */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={handleImageSelect}  // ✅ CHANGE THIS (was console.log)
        />
        {/* Plus button (for attachments) - WhatsApp style */}
        <button
          className="flex-shrink-0 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
          title="Attach"
          onClick={() => fileInputRef.current?.click()}  // added for img
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
        </button>

        {/* Text input - WhatsApp style */}
        <textarea
          className="flex-1 max-h-[120px] py-2 px-3 bg-transparent resize-none placeholder:text-gray-500 focus:outline-none text-sm leading-5"
          ref={textareaRef}
          rows="1"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message"
          dir="auto"
        />

        {/* Send button or microphone - WhatsApp style */}
        {(newMessage.trim() || selectedImage) ? (
          <button
            className="flex-shrink-0 p-2 text-[#25D366] hover:bg-gray-100 rounded-full transition-colors"
            onClick={handleSubmit}
            disabled={isLoading}
            title="Send"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        ) : (
          <button
            className="flex-shrink-0 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
            title="Record voice message"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

export default ChatInput;
