// frontend/src/pages/AuthPage.js
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthGoogle from '../components/AuthGoogle';
import { useAuth } from '../context/AuthContext';
import axiosInstance from '../api/axiosInstance'; // 백엔드 호출용

const AuthPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { login, isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/chat');
      return;
    }

    const searchParams = new URLSearchParams(location.search);
    const token = searchParams.get('access_token');
    const tokenType = searchParams.get('token_type');
    const userJson = searchParams.get('user'); // 백엔드가 user 정보를 query param으로 보내준다고 가정

    if (token && tokenType) {
      try {
        const userData = userJson ? JSON.parse(decodeURIComponent(userJson)) : null;
        login(token, userData);
        navigate('/chat');
      } catch (e) {
        console.error("Failed to parse user data or login", e);
        alert("로그인 처리 중 오류가 발생했습니다. 다시 시도해주세요.");
        navigate('/auth'); // 에러 발생 시 로그인 페이지로
      }
    }
  }, [location, navigate, login, isAuthenticated]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        backgroundColor: '#f0f2f5',
        gap: '20px',
      }}
    >
      <h1>AI Agent 챗봇에 로그인</h1>
      <AuthGoogle />
      <p>환영합니다! 서비스를 이용하시려면 Google 계정으로 로그인해주세요.</p>
    </div>
  );
};

export default AuthPage;