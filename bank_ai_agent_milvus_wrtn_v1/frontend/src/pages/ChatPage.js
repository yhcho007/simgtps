// frontend/src/pages/ChatPage.js
import React, { useRef, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import useChatSession from '../hooks/useChatSession';
import axiosInstance from '../api/axiosInstance';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput'; // 메시지 입력 컴포넌트 (하단에 추가)

const ChatPage = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const { sessionId, messages, addMessage, updateMessageFeedback } = useChatSession();
  const messagesEndRef = useRef(null);
  const [isSending, setIsSending] = useState(false); // 메시지 전송 중 상태

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/auth');
    }
  }, [isAuthenticated, navigate]);

  // 메시지가 추가될 때마다 스크롤을 맨 아래로
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (input) => {
    if (!input.trim() || !sessionId || isSending) return;

    // 사용자 메시지를 먼저 화면에 추가
    addMessage('user', input);
    setIsSending(true);

    try {
      const response = await axiosInstance.post('/agent/ask', {
        query: input,
        session_id: sessionId,
        customer_id: user?.id || 'anonymous', // 로그인된 사용자 ID 활용
        top_k_faq: 3,
      });

      const agentResponse = response.data;

      // Agent 메시지를 화면에 추가, 백엔드에서 받은 message_id 포함
      addMessage(
        'agent',
        agentResponse.text_response,
        agentResponse.multimodal_content,
        agentResponse.session_id, // 백엔드에서 확정된 세션 ID (필요시 사용)
        agentResponse.message_id // 백엔드에서 생성된 메시지 ID
      );

      console.log('Agent Response:', agentResponse);
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('agent', '죄송합니다. 메시지를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setIsSending(false);
    }
  };

  // 피드백 전송 후 UI 업데이트 핸들러
  const handleFeedbackUpdate = (messageId, feedbackType) => {
    updateMessageFeedback(messageId, feedbackType);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        maxWidth: '800px',
        margin: '0 auto',
        border: '1px solid #ccc',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    >
      <header
        style={{
          backgroundColor: '#f0f0f0',
          padding: '15px',
          borderBottom: '1px solid #eee',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <h2>AI 은행 챗봇</h2>
        <div>
          <span>{user?.name || user?.email}님</span>
          <button onClick={logout} style={{ marginLeft: '10px', padding: '5px 10px' }}>
            로그아웃
          </button>
        </div>
      </header>

      <main
        style={{
          flexGrow: 1,
          padding: '15px',
          overflowY: 'auto',
          backgroundColor: '#fff',
        }}
      >
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} onFeedback={handleFeedbackUpdate} />
        ))}
        <div ref={messagesEndRef} />
        {isSending && (
          <div style={{ textAlign: 'center', padding: '10px', color: '#888' }}>
            답변 생성 중...
          </div>
        )}
      </main>

      <footer
        style={{
          padding: '15px',
          borderTop: '1px solid #eee',
          backgroundColor: '#f0f0f0',
        }}
      >
        <ChatInput onSendMessage={handleSendMessage} disabled={isSending} />
      </footer>
    </div>
  );
};

export default ChatPage;