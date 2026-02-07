import { useEffect, useRef, useState, useCallback } from 'react';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE_URL = BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

/**
 * Custom hook for WebSocket connections
 * @param {string} url - WebSocket URL path (e.g., '/ws/user/{session_id}')
 * @param {object} options - Options object
 * @param {function} options.onMessage - Callback for incoming messages
 * @param {function} options.onError - Callback for errors
 * @param {function} options.onOpen - Callback when connection opens
 * @param {function} options.onClose - Callback when connection closes
 * @param {boolean} options.enabled - Whether to connect (default: true)
 * @returns {object} - { connected, sendMessage, reconnect }
 */
export function useWebSocket(url, options = {}) {
  const {
    onMessage,
    onError,
    onOpen,
    onClose,
    enabled = true
  } = options;

  // Use refs for callbacks to prevent reconnection loops
  const onMessageRef = useRef(onMessage);
  const onErrorRef = useRef(onError);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage;
    onErrorRef.current = onError;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
  }, [onMessage, onError, onOpen, onClose]);

  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds
  const urlRef = useRef(url);
  const enabledRef = useRef(enabled);

  // Update refs when values change
  useEffect(() => {
    urlRef.current = url;
    enabledRef.current = enabled;
  }, [url, enabled]);

  const connect = useCallback(() => {
    // Don't connect if already connected or if disabled
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[WebSocket] Already connected, skipping');
      return;
    }

    if (!enabledRef.current || !urlRef.current || urlRef.current === 'null' || urlRef.current === 'undefined') {
      console.log('[WebSocket] Connection disabled or no URL');
      return;
    }

    // Disconnect existing connection if any
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) {
        // Ignore errors when closing
      }
      wsRef.current = null;
    }

    try {
      const wsUrl = `${WS_BASE_URL}${urlRef.current}`;
      console.log('[WebSocket] Connecting to:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        setConnected(true);
        reconnectAttemptsRef.current = 0;
        if (onOpenRef.current) onOpenRef.current();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('[WebSocket] Message received:', data);
          if (onMessageRef.current) onMessageRef.current(data);
        } catch (e) {
          console.error('[WebSocket] Error parsing message:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        if (onErrorRef.current) onErrorRef.current(error);
      };

      ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected:', event.code, event.reason);
        setConnected(false);
        if (onCloseRef.current) onCloseRef.current(event);

        // Clear the ref so we can reconnect
        wsRef.current = null;

        // Attempt to reconnect if not a normal closure and still enabled
        if (event.code !== 1000 && enabledRef.current && urlRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(`[WebSocket] Reconnecting (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        }
      };
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);
      if (onErrorRef.current) onErrorRef.current(error);
    }
  }, []); // No dependencies - uses refs instead

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    reconnectAttemptsRef.current = 0; // Reset reconnect attempts
    
    if (wsRef.current) {
      try {
        wsRef.current.close(1000, 'Manual disconnect');
      } catch (e) {
        // Ignore errors
      }
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        const data = typeof message === 'string' ? message : JSON.stringify(message);
        wsRef.current.send(data);
        return true;
      } catch (error) {
        console.error('[WebSocket] Error sending message:', error);
        return false;
      }
    } else {
      console.warn('[WebSocket] Cannot send message: not connected');
      return false;
    }
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(() => {
      connect();
    }, 100);
  }, [disconnect, connect]);

  // Main effect - only reconnect when url or enabled changes
  useEffect(() => {
    if (enabled && url) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, url]); // Only depend on url and enabled, not callbacks

  // Send ping every 30 seconds to keep connection alive
  useEffect(() => {
    if (!connected) return;

    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' });
    }, 30000);

    return () => clearInterval(pingInterval);
  }, [connected, sendMessage]);

  return {
    connected,
    sendMessage,
    reconnect,
    disconnect
  };
}

