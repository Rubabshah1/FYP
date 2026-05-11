import { useEffect, useRef } from 'react';

/**
 * Hook that triggers a callback after a period of inactivity.
 * @param {Function} onIdle - Callback to trigger when idle.
 * @param {number} timeout - Timeout in milliseconds (default 15 minutes).
 * @param {boolean} enabled - Whether the timer is enabled.
 */
export function useIdleTimer(onIdle, timeout = 15 * 60 * 1000, enabled = true) {
  const timerRef = useRef(null);
  const onIdleRef = useRef(onIdle);

  // Update ref so timer always uses latest callback without re-running effect
  useEffect(() => {
    onIdleRef.current = onIdle;
  }, [onIdle]);

  useEffect(() => {
    if (!enabled) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      return;
    }

    const resetTimer = () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      timerRef.current = setTimeout(() => {
        onIdleRef.current();
      }, timeout);
    };

    const events = [
      'mousedown',
      'mousemove',
      'keypress',
      'scroll',
      'touchstart',
      'click'
    ];

    const handleActivity = () => {
      resetTimer();
    };

    // Initialize timer
    resetTimer();

    // Add event listeners
    events.forEach(event => {
      window.addEventListener(event, handleActivity);
    });

    // Cleanup
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      events.forEach(event => {
        window.removeEventListener(event, handleActivity);
      });
    };
  }, [enabled, timeout]);
}
