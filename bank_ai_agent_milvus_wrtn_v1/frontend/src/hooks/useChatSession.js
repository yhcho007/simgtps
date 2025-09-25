// frontend/src/hooks/useChatSession.js
import { useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid'; // npm install uuid

const useChatSession = () => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]); // {id: "msg_uuid", role: "user/agent", content: "text", multimodal: {}, feedback: null, timestamp: "..."}

  useEffect(() => {
    // 세션 ID가 없으면 새로 생성 (또는 기존 세션 ID 복원 로직 추가 가능)
    let currentSessionId = localStorage.getItem('chat_session_id');
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      localStorage.setItem('chat_session_id', currentSessionId);
    }
    setSessionId(currentSessionId);

    // TODO: 백엔드에서 해당 sessionId의 기존 채팅 기록을 로드하는 로직 추가
    // (SessionManager가 DB에서 세션 기록을 가져오므로, 로그인 후 이 기록을 가져와야 함)
    // 예: axiosInstance.get(`/session/history?session_id=${currentSessionId}`)
  }, []);

  const addMessage = useCallback((role, content, multimodalContent = null) => {
    const newMessage = {
      id: uuidv4(), // 각 메시지마다 고유 ID 부여 (피드백용)
      role,
      content,
      multimodal: multimodalContent,
      feedback: null, // 초기 피드백 상태
      timestamp: new Date().toISOString(),
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    return newMessage.id; // 생성된 메시지 ID 반환
  }, []);

  const updateMessageFeedback = useCallback((messageId, feedbackType) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) =>
        msg.id === messageId ? { ...msg, feedback: feedbackType } : msg
      )
    );
  }, []);

  return { sessionId, messages, addMessage, updateMessageFeedback };
};

export default useChatSession;