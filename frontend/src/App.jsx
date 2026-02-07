import { useState, useEffect } from 'react';
import Chatbot from '@/components/Chatbot';
import AgentDashboard from '@/pages/AgentDashboard';
import AdminDashboard from '@/pages/AdminDashboard';
import UserLogin from '@/pages/UserLogin';
import AgentLogin from '@/pages/AgentLogin';
import AdminLogin from '@/pages/AdminLogin';
import Welcome from '@/pages/Welcome';
import logo from '@/assets/images/alkhidmat.png';

function App() {
  const [view, setView] = useState(null); // null = welcome page, 'user', 'agent', 'admin'
  const [isAuthenticated, setIsAuthenticated] = useState({
    user: false,
    agent: false,
    admin: false
  });
  const [userSession, setUserSession] = useState(null);
  const [showWelcome, setShowWelcome] = useState(true);

  useEffect(() => {
    // Check authentication status
    const userSessionId = localStorage.getItem('user_session_id');
    const agentToken = localStorage.getItem('agent_token');
    const adminToken = localStorage.getItem('admin_token');

    const authState = {
      user: !!userSessionId,
      agent: !!agentToken,
      admin: !!adminToken
    };
    setIsAuthenticated(authState);

    // Check URL for routing
    const hash = window.location.hash;
    if (hash === '#/agent' || hash.includes('agent')) {
      setView('agent');
      setShowWelcome(false);
    } else if (hash === '#/admin' || hash.includes('admin')) {
      setView('admin');
      setShowWelcome(false);
    } else if (hash === '#/user' || hash.includes('user')) {
      setView('user');
      setShowWelcome(false);
    } else if (authState.user || authState.agent || authState.admin) {
      // If already authenticated, go to appropriate view
      if (authState.user) {
        setView('user');
        setShowWelcome(false);
      } else if (authState.agent) {
        setView('agent');
        setShowWelcome(false);
      } else if (authState.admin) {
        setView('admin');
        setShowWelcome(false);
      }
    } else {
      // Show welcome page if not authenticated
      setShowWelcome(true);
      setView(null);
    }
  }, []);

  function handleSelectUserType(type) {
    setView(type);
    setShowWelcome(false);
    window.location.hash = `#/${type}`;
  }

  function handleUserLogin(sessionData) {
    setUserSession(sessionData);
    setIsAuthenticated(prev => ({ ...prev, user: true }));
    setView('user');
    setShowWelcome(false);
  }

  function handleAgentLogin(agentData) {
    setIsAuthenticated(prev => ({ ...prev, agent: true }));
    setView('agent');
    setShowWelcome(false);
    window.location.hash = '#/agent';
  }

  function handleAdminLogin(adminData) {
    setIsAuthenticated(prev => ({ ...prev, admin: true }));
    setView('admin');
    setShowWelcome(false);
    window.location.hash = '#/admin';
  }

  // Show welcome page if not authenticated and no view selected
  if (showWelcome && !isAuthenticated.user && !isAuthenticated.agent && !isAuthenticated.admin) {
    return <Welcome onSelectUserType={handleSelectUserType} />;
  }

  // User view - requires login
  if (view === 'user') {
    if (!isAuthenticated.user) {
      return <UserLogin onLoginSuccess={handleUserLogin} onBack={() => {
        setShowWelcome(true);
        setView(null);
        window.location.hash = '';
      }} />;
    }
    
    function handleBackToWelcome() {
      localStorage.removeItem('user_session_id');
      localStorage.removeItem('user_id');
      setIsAuthenticated(prev => ({ ...prev, user: false }));
      setShowWelcome(true);
      setView(null);
      window.location.hash = '';
    }

    return (
      <div className='flex flex-col h-screen w-full max-w-3xl mx-auto bg-white'>
        {/* Chat area - full height with WhatsApp-style header inside Chatbot */}
        <div className='flex-1 overflow-hidden'>
          <Chatbot />
        </div>
      </div>
    );
  }

  // Agent view - requires login
  if (view === 'agent') {
    if (!isAuthenticated.agent) {
      return <AgentLogin onLoginSuccess={handleAgentLogin} onBack={() => {
        setShowWelcome(true);
        setView(null);
        window.location.hash = '';
      }} />;
    }
    return <AgentDashboard />;
  }

  // Admin view - requires login
  if (view === 'admin') {
    if (!isAuthenticated.admin) {
      return <AdminLogin onLoginSuccess={handleAdminLogin} onBack={() => {
        setShowWelcome(true);
        setView(null);
        window.location.hash = '';
      }} />;
    }
    return <AdminDashboard />;
  }

  // Default: show welcome page
  return <Welcome onSelectUserType={handleSelectUserType} />;
}

export default App;
