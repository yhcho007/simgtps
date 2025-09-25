// frontend/src/components/AuthGoogle.js
import React from 'react';
import axiosInstance from '../api/axiosInstance'; // 백엔드 호출용

const AuthGoogle = () => {
  const handleGoogleLogin = async () => {
    try {
      // 백엔드의 Google 로그인 시작 API를 호출
      // 이 API는 Google OAuth 페이지로 리다이렉트 응답을 보냄
      window.location.href = `${axiosInstance.defaults.baseURL}/auth/google/login`;
    } catch (error) {
      console.error("Google login initiation failed", error);
      alert("Google 로그인 시작에 실패했습니다. 다시 시도해주세요.");
    }
  };

  return (
    <button
      onClick={handleGoogleLogin}
      style={{
        padding: '10px 20px',
        fontSize: '16px',
        backgroundColor: '#4285F4',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
      }}
    >
      <img
        src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_2013_Google.png" // Google 로고 (임시)
        alt="Google logo"
        style={{ width: '20px', height: '20px' }}
      />
      Google 계정으로 로그인
    </button>
  );
};

export default AuthGoogle;