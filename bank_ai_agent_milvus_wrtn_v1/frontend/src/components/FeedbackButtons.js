// frontend/src/components/FeedbackButtons.js
import React, { useState } from 'react';
import axiosInstance from '../api/axiosInstance';

const FeedbackButtons = ({ messageId, sessionId, currentFeedback, onFeedback }) => {
  const [feedbackStatus, setFeedbackStatus] = useState(currentFeedback);
  const [loading, setLoading] = useState(false);

  const handleSubmitFeedback = async (feedbackType) => {
    if (loading || feedbackStatus === feedbackType) return; // 이미 피드백했거나 로딩 중이면 방지

    setLoading(true);
    try {
      await axiosInstance.post('/feedback/submit', {
        session_id: sessionId,
        message_id: messageId,
        feedback_type: feedbackType,
      });
      setFeedbackStatus(feedbackType);
      onFeedback(messageId, feedbackType); // 부모 컴포넌트에 상태 변경 알림
      console.log(`Feedback '${feedbackType}' submitted for message ID: ${messageId}`);
    } catch (error) {
      console.error('Failed to submit feedback', error);
      alert('피드백 전송에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '5px' }}>
      <button
        onClick={() => handleSubmitFeedback('like')}
        disabled={loading}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '1.2em',
          color: feedbackStatus === 'like' ? '#28a745' : '#888', // 좋아요 선택 시 색상 변경
        }}
      >
        👍 {loading && feedbackStatus === 'like' ? '전송 중...' : ''}
      </button>
      <button
        onClick={() => handleSubmitFeedback('dislike')}
        disabled={loading}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '1.2em',
          color: feedbackStatus === 'dislike' ? '#dc3545' : '#888', // 싫어요 선택 시 색상 변경
        }}
      >
        👎 {loading && feedbackStatus === 'dislike' ? '전송 중...' : ''}
      </button>
    </div>
  );
};

export default FeedbackButtons;