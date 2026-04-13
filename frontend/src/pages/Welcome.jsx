import { useState } from 'react';
import logo from '@/assets/images/alkhidmat.png';

function Welcome({ onSelectUserType }) {
  const [selectedType, setSelectedType] = useState(null);

  function handleSelect(type) {
    setSelectedType(type);
    // Small delay for visual feedback
    setTimeout(() => {
      onSelectUserType(type);
    }, 200);
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 px-4">
      <div className="max-w-4xl w-full">
        {/* Logo and Header */}
        <div className="text-center mb-12">
          <img src={logo} alt="Alkhidmat" className="w-40 mx-auto mb-6" />
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Welcome to Alkhidmat</h1>
          <p className="text-lg text-gray-600">Choose your login option to continue</p>
        </div>

        {/* Login Options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* User Option */}
          <div
            onClick={() => handleSelect('user')}
            className={`bg-white rounded-2xl shadow-lg p-8 cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-xl ${
              selectedType === 'user' ? 'ring-4 ring-blue-500 scale-105' : ''
            }`}
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">User</h3>
              <p className="text-sm text-gray-600 mb-4">
                Chat with our AI assistant or connect with a human agent
              </p>
              <button className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors">
                Continue as User
            </button>
            </div>
          </div>

          {/* Agent Option */}
          <div
            onClick={() => handleSelect('agent')}
            className={`bg-white rounded-2xl shadow-lg p-8 cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-xl ${
              selectedType === 'agent' ? 'ring-4 ring-yellow-500 scale-105' : ''
            }`}
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Agent</h3>
              <p className="text-sm text-gray-600 mb-4">
                Access the agent dashboard to manage and resolve tickets
              </p>
              <button className="w-full py-3 bg-yellow-600 text-white rounded-lg font-medium hover:bg-yellow-700 transition-colors">
                Agent Login
              </button>
            </div>
          </div>

          {/* Admin Option */}
                <div
            onClick={() => handleSelect('admin')}
            className={`bg-white rounded-2xl shadow-lg p-8 cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-xl ${
              selectedType === 'admin' ? 'ring-4 ring-blue-500 scale-105' : ''
                  }`}
                >
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                      </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Admin</h3>
              <p className="text-sm text-gray-600 mb-4">
                View analytics and manage the entire support system
              </p>
              <button className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors">
                Admin Login
              </button>
                </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="text-center mt-12">
          <p className="text-sm text-gray-500">
            Need help? Contact support at{' '}
            <a href="mailto:info@alkhidmat.org" className="text-blue-600 hover:text-blue-800">
              info@alkhidmat.org
            </a>
          </p>
          </div>
      </div>
    </div>
  );
}

export default Welcome;
