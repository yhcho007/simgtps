// frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import AuthPage from './pages/AuthPage';
import ChatPage from './pages/ChatPage';
import './index.css'; // 전역 스타일

// 로그인 상태를 확인하여 접근을 제한하는 ProtectedRoute
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/auth" />;
};

const App = () => {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/auth" element={<AuthPage />} />
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <ChatPage />
              </ProtectedRoute>
            }
          />
          {/* 기본 경로를 로그인 페이지로 리다이렉트 */}
          <Route path="/" element={<Navigate to="/chat" />} />
          <Route path="*" element={<p>404 Not Found</p>} /> {/* 404 페이지 */}
        </Routes>
      </Router>
    </AuthProvider>
  );
};

export default App;