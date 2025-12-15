# Alkhidmat Chat Portal - Frontend

React frontend application for the Alkhidmat Chat Portal.

## Prerequisites

- Node.js 18+ and npm

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Frontend

### Development Mode

```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Environment Variables

Create a `.env` file in the frontend directory (optional):

```env
VITE_API_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`.

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   │   ├── Chatbot.jsx
│   │   ├── ChatMessages.jsx
│   │   ├── ChatInput.jsx
│   │   └── Spinner.jsx
│   ├── pages/          # Page components
│   │   ├── Welcome.jsx
│   │   ├── UserLogin.jsx
│   │   ├── AgentLogin.jsx
│   │   ├── AdminLogin.jsx
│   │   ├── AgentDashboard.jsx
│   │   └── AdminDashboard.jsx
│   ├── hooks/          # Custom React hooks
│   ├── assets/         # Images and static assets
│   ├── api.js          # API client
│   ├── App.jsx         # Main app component
│   └── main.jsx        # Entry point
├── public/             # Public assets
├── package.json
└── vite.config.js      # Vite configuration
```

## Features

- **User Interface**: OTP-based login, chat with RAG AI
- **Agent Dashboard**: View and manage tickets, chat with users
- **Admin Dashboard**: Analytics and ticket monitoring

## Path Aliases

The project uses `@/` as an alias for `src/` directory:
- `@/components` → `src/components`
- `@/pages` → `src/pages`
- `@/assets` → `src/assets`

Configured in `vite.config.js` and `jsconfig.json`.

