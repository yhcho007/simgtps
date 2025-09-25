// frontend/src/context/AuthContext.js
import React, { createContext, useState, useEffect, useContext } from 'react';
import axiosInstance from '../api/axiosInstance'; // JWT 토큰을 자동으로 헤더에 포함

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [jwtToken, setJwtToken] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('jwt_token');
    const storedUser = localStorage.getItem('user_info');
    if (token && storedUser) {
      try {
        const userInfo = JSON.parse(storedUser);
        setJwtToken(token);
        setUser(userInfo);
        setIsAuthenticated(true);
        console.log("Existing JWT token and user info loaded.");
      } catch (e) {
        console.error("Failed to parse user info from localStorage", e);
        logout(); // 유효하지 않은 정보는 제거
      }
    } else {
      setIsAuthenticated(false);
      setUser(null);
      setJwtToken(null);
    }
  }, []);

  const login = (token, userData) => {
    localStorage.setItem('jwt_token', token);
    localStorage.setItem('user_info', JSON.stringify(userData));
    setJwtToken(token);
    setUser(userData);
    setIsAuthenticated(true);
  };

  const logout = () => {
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('user_info');
    setJwtToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, jwtToken, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);