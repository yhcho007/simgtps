// client/src/App.js
import React, { useState, useEffect } from 'react';
import './App.css'; // 기본 CSS 파일 불러오기

function App() {
  const [messages, setMessages] = useState([]); // 채팅 메시지들을 저장할 상태
  const [inputMessage, setInputMessage] = useState(''); // 사용자가 입력할 메시지 상태
  const [isLoading, setIsLoading] = useState(false); // 로딩 중인지 여부

  // 메시지가 추가될 때마다 스크롤을 맨 아래로 이동
  useEffect(() => {
    const chatContainer = document.getElementById('chat-container');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }, [messages]);

  // 메시지 전송 함수
  const sendMessage = async () => {
    if (inputMessage.trim() === '' || isLoading) return; // 빈 메시지이거나 로딩 중이면 전송 안 함

    const newMessage = { sender: 'user', text: inputMessage };
    setMessages((prevMessages) => [...prevMessages, newMessage]); // 사용자 메시지를 채팅에 추가
    setInputMessage(''); // 입력창 비우기
    setIsLoading(true); // 로딩 시작

    try {
      // 백엔드 API 호출!
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }), // 사용자 메시지를 JSON 형태로 백엔드에 보냄
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json(); // 백엔드로부터 받은 응답을 JSON으로 파싱
      const botReply = { sender: 'bot', text: data.reply }; // 봇의 답변
      setMessages((prevMessages) => [...prevMessages, botReply]); // 봇의 답변을 채팅에 추가
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { sender: 'bot', text: '죄송해요. 메시지를 보내는 데 문제가 생겼어요. ㅠㅠ' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false); // 로딩 끝
    }
  };

  // 엔터 키 입력 시 메시지 전송
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>나만의 챗봇 ✨</h1>
      </header>
      <div className="chat-window">
        <div id="chat-container" className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="message-bubble">{msg.text}</div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-bubble">생각 중... 🤖</div>
            </div>
          )}
        </div>
        <div className="chat-input-area">
          <input
            type="text"
            placeholder="메시지를 입력하세요..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
          />
          <button onClick={sendMessage} disabled={isLoading}>
            보내기
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;