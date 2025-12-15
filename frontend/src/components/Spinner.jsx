function Spinner() {
  return (
    <div className='flex items-center gap-1 py-2'>
      <span className='typing-dot' />
      <span className='typing-dot typing-dot-delay-1' />
      <span className='typing-dot typing-dot-delay-2' />
    </div>
  );
}

export default Spinner;