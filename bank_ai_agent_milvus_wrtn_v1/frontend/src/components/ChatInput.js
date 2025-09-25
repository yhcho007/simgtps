// frontend/src/components/ChatInput.js
import React, { useState } from 'react';

const ChatInput = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="메시지를 입력하세요..."
        disabled={disabled}
        style={{
          flexGrow: 1,
          padding: '10px',
          border: '1px solid #ccc',
          borderRadius: '5px',
          fontSize: '16px',
        }}
      />
      <button
        type="submit"
        disabled={disabled}
        style={{
          padding: '10px 20px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.7 : 1,
        }}
      >
        전송
      </button>
    </form>
  );
};

export default ChatInput;