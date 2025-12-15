import { useState, useEffect, useRef, useCallback } from 'react';
import api from '@/api';
import logo from '@/assets/images/alkhidmat.png';

function AdminDashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, active, in_progress, resolved
  const loadingRef = useRef(false); // Prevent concurrent requests

  const loadData = useCallback(async () => {
    if (loadingRef.current) return; // Skip if already loading
    loadingRef.current = true;
    try {
      const [analyticsData, ticketsData] = await Promise.all([
        api.getAnalytics(),
        api.adminListTickets(filter === 'all' ? null : filter)
      ]);
      setAnalytics(analyticsData);
      setTickets(ticketsData.tickets || []);
    } catch (err) {
      console.error('Failed to load data:', err);
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, [filter]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 15000); // Reduced to 15 seconds (was 10)
    return () => clearInterval(interval);
  }, [loadData]);

  function formatTime(seconds) {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  }

  function handleLogout() {
    localStorage.removeItem('admin_token');
    localStorage.removeItem('admin_id');
    window.location.reload();
  }

  if (loading && !analytics) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <img src={logo} alt="Alkhidmat" className="w-12 h-12" />
              <div>
                <h1 className="text-2xl font-bold text-gray-800">Admin Dashboard</h1>
                <p className="text-sm text-gray-500">Alkhidmat Support Portal</p>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  localStorage.removeItem('admin_token');
                  localStorage.removeItem('admin_id');
                  window.location.hash = '';
                  window.location.reload();
                }}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm font-medium"
              >
                ← Back to Welcome
              </button>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Analytics Cards */}
        {analytics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-500 mb-1">Total Tickets</div>
              <div className="text-3xl font-bold text-gray-800">{analytics.total_tickets}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-500 mb-1">Active Tickets</div>
              <div className="text-3xl font-bold text-yellow-600">{analytics.active_tickets}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-500 mb-1">In Progress</div>
              <div className="text-3xl font-bold text-blue-600">{analytics.in_progress_tickets}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-500 mb-1">Resolved</div>
              <div className="text-3xl font-bold text-green-600">{analytics.resolved_tickets}</div>
            </div>
          </div>
        )}

        {/* Average Resolution Time */}
        {analytics && analytics.average_resolution_time_seconds && (
          <div className="bg-white rounded-lg shadow p-6 mb-8">
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Performance Metrics</h2>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-gray-800">
                {formatTime(analytics.average_resolution_time_seconds)}
              </span>
              <span className="text-gray-500">Average Resolution Time</span>
            </div>
          </div>
        )}

        {/* Tickets Section */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-800">All Tickets</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setFilter('all')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filter === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  All
                </button>
                <button
                  onClick={() => setFilter('active')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filter === 'active' ? 'bg-yellow-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Active
                </button>
                <button
                  onClick={() => setFilter('in_progress')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filter === 'in_progress' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  In Progress
                </button>
                <button
                  onClick={() => setFilter('resolved')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filter === 'resolved' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Resolved
                </button>
              </div>
            </div>
          </div>

          <div className="divide-y divide-gray-200">
            {tickets.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                No tickets found
              </div>
            ) : (
              tickets.map(ticket => (
                <div
                  key={ticket.ticket_id}
                  onClick={() => setSelectedTicket(ticket)}
                  className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                    selectedTicket?.ticket_id === ticket.ticket_id ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-semibold text-gray-800">#{ticket.ticket_id.slice(0, 8)}</span>
                        <span className={`px-2 py-0.5 text-xs rounded-full ${
                          ticket.status === 'active' ? 'bg-yellow-100 text-yellow-800' :
                          ticket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {ticket.status}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mb-1">
                        Created: {new Date(ticket.created_at).toLocaleString()}
                      </p>
                      {ticket.resolved_at && (
                        <p className="text-xs text-gray-500">
                          Resolved: {new Date(ticket.resolved_at).toLocaleString()}
                        </p>
                      )}
                      {ticket.response?.content && (
                        <p className="text-sm text-gray-700 mt-2 line-clamp-2">
                          {ticket.response.content}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Ticket Detail Modal */}
        {selectedTicket && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                <h3 className="text-xl font-semibold text-gray-800">
                  Ticket #{selectedTicket.ticket_id.slice(0, 8)}
                </h3>
                <button
                  onClick={() => setSelectedTicket(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Status</label>
                  <div className="mt-1">
                    <span className={`px-3 py-1 text-sm rounded-full ${
                      selectedTicket.status === 'active' ? 'bg-yellow-100 text-yellow-800' :
                      selectedTicket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {selectedTicket.status}
                    </span>
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Created At</label>
                  <p className="mt-1 text-gray-800">{new Date(selectedTicket.created_at).toLocaleString()}</p>
                </div>
                {selectedTicket.resolved_at && (
                  <div>
                    <label className="text-sm font-medium text-gray-500">Resolved At</label>
                    <p className="mt-1 text-gray-800">{new Date(selectedTicket.resolved_at).toLocaleString()}</p>
                  </div>
                )}
                {selectedTicket.agent_id && (
                  <div>
                    <label className="text-sm font-medium text-gray-500">Assigned Agent</label>
                    <p className="mt-1 text-gray-800">{selectedTicket.agent_id}</p>
                  </div>
                )}
                {selectedTicket.response?.content && (
                  <div>
                    <label className="text-sm font-medium text-gray-500">Initial Message</label>
                    <p className="mt-1 text-gray-800">{selectedTicket.response.content}</p>
                  </div>
                )}
                {selectedTicket.session && (
                  <div>
                    <label className="text-sm font-medium text-gray-500">Session Info</label>
                    <p className="mt-1 text-gray-800">Session ID: {selectedTicket.session.session_id}</p>
                    <p className="text-sm text-gray-500">User ID: {selectedTicket.session.user_id}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AdminDashboard;

