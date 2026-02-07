import { useEffect, useLayoutEffect, useRef } from 'react';

const SCROLL_THRESHOLD = 50; // Increased threshold for better UX

function useAutoScroll(messages, isLoading) {
  const scrollContentRef = useRef(null);
  const isDisabled = useRef(false);
  const prevScrollTop = useRef(null);
  const shouldAutoScroll = useRef(true);

  // Scroll to bottom function
  const scrollToBottom = (behavior = 'smooth') => {
    if (scrollContentRef.current) {
      const container = scrollContentRef.current;
      container.scrollTo({
        top: container.scrollHeight,
        behavior
      });
    }
  };

  // Scroll to bottom when messages change or when loading starts/stops
  useLayoutEffect(() => {
    if (!scrollContentRef.current) return;

    // Always scroll to bottom when loading (new message being generated)
    if (isLoading) {
      shouldAutoScroll.current = true;
      scrollToBottom('smooth');
      return;
    }

    // If auto-scroll is enabled, scroll to bottom when messages change
    if (shouldAutoScroll.current && messages.length > 0) {
      scrollToBottom('smooth');
    }
  }, [messages, isLoading]);

  // Handle scroll events to detect user scrolling up
  useEffect(() => {
    const container = scrollContentRef.current;
    if (!container) return;

    function onScroll() {
      if (!container) return;
      
      const { scrollHeight, clientHeight, scrollTop } = container;
      const isNearBottom = scrollHeight - clientHeight - scrollTop <= SCROLL_THRESHOLD;
      
      // If user scrolls near bottom, re-enable auto-scroll
      if (isNearBottom) {
        shouldAutoScroll.current = true;
        isDisabled.current = false;
      } else if (prevScrollTop.current !== null && scrollTop < prevScrollTop.current) {
        // User scrolled up, disable auto-scroll
        shouldAutoScroll.current = false;
        isDisabled.current = true;
      }
      
      prevScrollTop.current = scrollTop;
    }

    container.addEventListener('scroll', onScroll, { passive: true });
    
    // Initialize scroll position
    prevScrollTop.current = container.scrollTop;
    
    return () => {
      container.removeEventListener('scroll', onScroll);
    };
  }, []);

  // Handle resize events (e.g., window resize, content changes)
  useEffect(() => {
    const container = scrollContentRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      if (shouldAutoScroll.current && !isDisabled.current) {
        scrollToBottom('smooth');
      }
    });

    resizeObserver.observe(container);
    
    return () => resizeObserver.disconnect();
  }, []);

  // Initial scroll to bottom when component mounts
  useEffect(() => {
    if (scrollContentRef.current && messages.length > 0) {
      scrollToBottom('auto'); // Instant scroll on mount
    }
  }, []); // Only run once on mount

  return scrollContentRef;
}

export default useAutoScroll;