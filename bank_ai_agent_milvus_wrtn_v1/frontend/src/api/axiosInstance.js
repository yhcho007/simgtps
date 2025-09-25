// frontend/src/api/axiosInstance.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // 백엔드 API 주소

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터: 로컬 스토리지에서 JWT 토큰을 가져와 Authorization 헤더에 추가
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('jwt_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터: 401 Unauthorized 에러 발생 시 로그아웃 처리
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      console.error("401 Unauthorized: JWT 토큰 만료 또는 유효하지 않음.");
      // 여기에 로그아웃 처리 로직 추가 (예: localStorage.removeItem('jwt_token'); window.location.href = '/login';)
      localStorage.removeItem('jwt_token');
      window.location.href = '/auth'; // 로그인 페이지로 리다이렉트
    }
    return Promise.reject(error);
  }
);

export default axiosInstance;