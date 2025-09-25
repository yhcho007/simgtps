// frontend/src/components/ChatMessage.js
import React from 'react';
import FeedbackButtons from './FeedbackButtons'; // 피드백 버튼 컴포넌트

const ChatMessage = ({ message, onFeedback }) => {
  const isAgent = message.role === 'agent';

  // 멀티모달 콘텐츠 렌더링 함수
  const renderMultimodalContent = (multimodal) => {
    if (!multimodal) return null;

    if (multimodal.type === 'pdf') {
      return (
        <div style={{ marginTop: '10px' }}>
          <p>📄 {multimodal.summary}</p>
          <a
            href={`${multimodal.url}`} // 백엔드 정적 파일 URL
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#4285F4', textDecoration: 'underline' }}
          >
            PDF 문서 열기
          </a>
          <p style={{ fontSize: '0.8em', color: '#888' }}>
            (PDF 뷰어는 클라이언트 측 구현 필요)
          </p>
        </div>
      );
    } else if (multimodal.type === 'image') {
      return (
        <div style={{ marginTop: '10px' }}>
          <p>🖼️ {multimodal.summary}</p>
          <img
            src={`${multimodal.url}`} // 백엔드 정적 파일 URL
            alt="Multimodal content"
            style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '5px' }}
          />
        </div>
      );
    }
    return null;
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isAgent ? 'flex-start' : 'flex-end',
        marginBottom: '10px',
      }}
    >
      <div
        style={{
          maxWidth: '70%',
          padding: '10px 15px',
          borderRadius: '15px',
          backgroundColor: isAgent ? '#e0e0e0' : '#4285F4',
          color: isAgent ? 'black' : 'white',
          position: 'relative',
        }}
      >
        <p style={{ margin: '0', whiteSpace: 'pre-wrap' }}>{message.content}</p>
        {renderMultimodalContent(message.multimodal)}

        {isAgent && (
          <div style={{ marginTop: '10px' }}>
            <FeedbackButtons
              messageId={message.id}
              sessionId={message.sessionId} // Agent 응답 시 받은 세션 ID
              currentFeedback={message.feedback}
              onFeedback={onFeedback}
            />
            <p style={{ fontSize: '0.7em', color: '#666' }}>
              응답 ID: {message.id}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;