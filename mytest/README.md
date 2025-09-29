헐 조윤희4305님! 😮 프론트엔드는 처음이라구요?! 끄덕끄덕, 완전 괜찮아요! 누구나 시작은 똑같답니다! 개발은 모르는 걸 배우고 만들어가는 재미 아니겠어용?! 조윤희4305님, 너무 걱정 마세요! 저 응이가 쉽고 친절하게, 그것도 딱! 조윤희4305님 맞춤으로 **React (프론트엔드) + Node.js (백엔드) 조합의 챗봇 만드는 법**을 차근차근 알려드릴게요! 이거 같이 하다 보면 "어? 생각보다 할 만하네?" 하실 거예요! 후훗 😎

조윤희4305님이 AI Agent랑 챗봇 UI를 연마하고 싶어 하시니, 이 과정이 진짜 알찬 경험이 될 거예요! 시작해볼까요?!

---

### **✨ 초보자를 위한 React + Node.js 챗봇 만들기! ✨**

일단 전체 그림부터 그려봐요! 챗봇은 크게 두 부분으로 나뉘어요:

1.  **프론트엔드 (Frontend - React):** 사용자가 보는 화면이에요! 채팅 입력창, 메시지 표시되는 곳 같은 UI를 만들죠. 조윤희4305님 앱 화면이 될 부분이에요.
2.  **백엔드 (Backend - Node.js):** 사용자가 입력한 메시지를 받아서 실제 AI(OpenAI 같은)랑 대화하고, AI의 답변을 받아서 다시 프론트엔드로 보내주는 뇌 역할을 해요.

우리는 이 두 개를 따로따로 만들고 서로 대화하게 할 거예요!

---

### **STEP 0: 준비물 챙기기! (설치)**

개발 시작 전에 몇 가지 프로그램들을 설치해야 해요!

1.  **Node.js 설치:**
    *   자바스크립트를 컴퓨터에서 실행시켜주는 환경이에요. 프론트엔드 React 개발과 백엔드 Node.js 개발 모두에 필요하니까 꼭 설치해야 해요!
    *   **설치 방법:** [Node.js 공식 홈페이지](https://nodejs.org/ko/download/) 에 가서 Lts 버전 (Recommended for Most Users)을 다운로드하고 설치해주세요. 그냥 계속 "Next" 누르시면 돼요!
    *   **설치 확인:** 터미널(또는 명령 프롬프트)을 열고 다음 명령어를 입력해서 버전이 나오면 성공!
        ```bash
        node -v
        npm -v
        ```
        `v18.x.x`나 `v20.x.x` 등 버전이 뜨면 됩니다!

2.  **코드 편집기 (Visual Studio Code 추천):**
    *   코딩하기 편한 프로그램이에요. VS Code가 제일 인기가 많고 좋아요!
    *   **설치 방법:** [VS Code 공식 홈페이지](https://code.visualstudio.com/) 에서 다운로드해서 설치해주세요.

---

### **STEP 1: 백엔드 만들기 (Node.js)**

백엔드는 `server.js`라는 파일을 만들어서 API 역할을 하게 할 거예요.

#### **1. 프로젝트 폴더 생성 및 초기화**

*   아무 폴더나 만들고 그 안으로 이동해서 `server` 폴더를 만들게요.
    ```bash
    mkdir my-chatbot-app # 전체 프로젝트 폴더
    cd my-chatbot-app
    mkdir server # 백엔드 폴더
    cd server
    npm init -y # package.json 파일 생성 (계속 yes)
    ```

#### **2. 필요한 라이브러리 설치**

*   `express`: 웹 서버를 쉽게 만들어주는 라이브러리예요.
*   `cors`: 프론트엔드와 백엔드가 다른 주소(port)로 실행될 때 통신할 수 있게 해줘요. (아니면 에러남!)
*   `openai`: (나중에 쓸 거지만 일단 같이 설치!) OpenAI API를 쉽게 쓸 수 있게 해주는 라이브러리.

```bash
npm install express cors openai dotenv
```
`dotenv`는 나중에 API Key 같은 민감한 정보를 안전하게 관리하기 위해 설치해요.

#### **3. 백엔드 코드 작성 (`server/server.js`)**

`server` 폴더 안에 `server.js` 파일을 만들고 아래 코드를 복사 붙여넣기 해주세요.

```javascript
// server/server.js
const express = require('express'); // express 라이브러리 불러오기
const cors = require('cors'); // cors 라이브러리 불러오기
const { OpenAI } = require('openai'); // openai 라이브러리에서 OpenAI 객체 불러오기
require('dotenv').config(); // .env 파일에서 환경 변수를 로드

const app = express(); // express 앱 생성
const port = 5000; // 백엔드 서버가 5000번 포트로 실행될 거예요

// CORS 설정: 모든 도메인에서의 요청을 허용 (개발 시 편의용, 실제 서비스에서는 보안 강화 필요!)
app.use(cors());

// JSON 형식의 요청 본문을 파싱할 수 있도록 설정
app.use(express.json());

// OpenAI API 키 설정
// .env 파일에 OPENAI_API_KEY=YOUR_API_KEY 형태로 저장해야 해요.
// 아직 OpenAI API 키가 없다면 일단 임시로 "안녕하세요!" 같은 메시지를 응답하도록 할게요.
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY, // process.env를 통해 .env 파일에서 키를 불러옴
});

// 테스트용 루트 경로 API (선택 사항)
app.get('/', (req, res) => {
    res.send('Chatbot Backend is running!');
});

// 챗봇과의 대화를 처리할 API 엔드포인트
app.post('/api/chat', async (req, res) => {
    // 챗봇 프론트엔드에서 보낸 메시지를 받아요.
    const userMessage = req.body.message; 
    console.log(`Received message: ${userMessage}`);

    // 여기에 실제 OpenAI API 호출 로직이 들어갈 거예요!
    // 지금은 예시로 고정된 답변을 주거나, 간단히 OpenAI API를 호출하는 코드를 넣어볼게요.
    try {
        if (!process.env.OPENAI_API_KEY) {
            // API 키가 없으면 임시 메시지 반환 (초보자용)
            console.warn("OpenAI API Key is not set. Responding with a dummy message.");
            res.json({ reply: `(백엔드: 안녕하세요! '${userMessage}'라고 말씀하셨군요!) 아직 OpenAI API 연동 전이라 제가 똑똑한 답변은 못 드리지만, 잘 받았습니다! 😊` });
            return;
        }

        const completion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo", // 사용하고 싶은 OpenAI 모델 지정 (예: gpt-4, gpt-4o 등)
            messages: [{ role: "user", content: userMessage }],
        });

        const botReply = completion.choices[0].message.content;
        console.log(`Bot reply: ${botReply}`);
        res.json({ reply: botReply }); // AI의 답변을 프론트엔드로 보내줘요.

    } catch (error) {
        console.error("Error calling OpenAI API:", error.message);
        res.status(500).json({ reply: "죄송해요, AI와 대화 중에 오류가 발생했어요. ㅠㅠ" });
    }
});

// 서버 시작!
app.listen(port, () => {
    console.log(`Backend server listening at http://localhost:${port}`);
    console.log('Ctrl + C 를 눌러 서버를 종료할 수 있습니다.');
});
```

#### **4. `.env` 파일 설정**

`server` 폴더 안에 `.env` 파일을 만들고 여기에 OpenAI API 키를 입력하세요. **`YOUR_OPENAI_API_KEY_HERE`** 부분에 조윤희4305님의 실제 OpenAI API 키를 넣어주세요. (OpenAI 홈페이지에서 발급받아야 해요!)
```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```
**⚠️주의:** `.env` 파일은 절대 GitHub 같은 공개 저장소에 올리면 안 돼요! 개인정보 유출 위험이 있어요!

#### **5. 백엔드 실행**

`server` 폴더 경로에서 터미널에 다음 명령어를 입력하세요.

```bash
node server.js
```

터미널에 `Backend server listening at http://localhost:5000` 이런 메시지가 뜨면 성공! 🥳

---

### **STEP 2: 프론트엔드 만들기 (React)**

이제 사용자들이 실제 사용할 화면을 React로 만들어볼게요!

#### **1. React 앱 생성**

*   `my-chatbot-app` (전체 프로젝트 폴더)으로 다시 이동해서 `client` 폴더에 React 앱을 만들게요.
    ```bash
    cd .. # my-chatbot-app 폴더로 이동
    npx create-react-app client # client 폴더에 React 앱 생성
    cd client
    ```
    *이 과정은 시간이 좀 걸릴 수 있어요. 커피 한 잔 마시고 오세요!* ☕

#### **2. 프론트엔드 코드 수정 (`client/src/App.js`)**

`client` 폴더 안 `src` 폴더에 `App.js` 파일을 열고, 기존 내용을 모두 지우고 아래 코드를 붙여넣기 해주세요.

```javascript
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
```

#### **3. 프론트엔드 스타일 (`client/src/App.css`)**

`client` 폴더 안 `src` 폴더에 `App.css` 파일을 열고, 기존 내용을 모두 지우고 아래 코드를 붙여넣기 해주세요. 이 코드는 챗봇 화면을 예쁘게 꾸며줄 거예요!

```css
/* client/src/App.css */
.App {
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background-color: #f0f2f5;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
}

.App-header {
  background-color: #61dafb;
  padding: 20px;
  width: 100%;
  max-width: 600px;
  border-radius: 8px 8px 0 0;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.App-header h1 {
  margin: 0;
  color: #fff;
  font-size: 2.5em;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}

.chat-window {
  background-color: #ffffff;
  width: 100%;
  max-width: 600px;
  height: 70vh; /* 채팅창 높이 */
  border-radius: 0 0 8px 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden; /* 메시지가 넘치면 스크롤 */
  border: 1px solid #ddd;
}

.chat-messages {
  flex-grow: 1; /* 남은 공간을 모두 차지 */
  padding: 20px;
  overflow-y: auto; /* 내용이 넘치면 스크롤바 생성 */
  background-color: #e9ebee;
  border-bottom: 1px solid #eee;
}

.message {
  display: flex;
  margin-bottom: 15px;
}

.message.user {
  justify-content: flex-end; /* 사용자 메시지는 오른쪽 정렬 */
}

.message.bot {
  justify-content: flex-start; /* 봇 메시지는 왼쪽 정렬 */
}

.message-bubble {
  max-width: 70%;
  padding: 12px 18px;
  border-radius: 20px;
  line-height: 1.5;
  word-wrap: break-word; /* 긴 단어 잘림 방지 */
}

.message.user .message-bubble {
  background-color: #007bff;
  color: white;
  border-bottom-right-radius: 5px; /* 끝부분 둥글기 줄임 */
}

.message.bot .message-bubble {
  background-color: #fefefe;
  color: #333;
  border: 1px solid #ddd;
  border-bottom-left-radius: 5px; /* 끝부분 둥글기 줄임 */
}

.chat-input-area {
  display: flex;
  padding: 15px 20px;
  border-top: 1px solid #eee;
  background-color: #fff;
}

.chat-input-area input {
  flex-grow: 1; /* 남은 공간 모두 차지 */
  border: 1px solid #ddd;
  border-radius: 20px;
  padding: 10px 15px;
  font-size: 1em;
  margin-right: 10px;
  transition: border-color 0.2s;
}

.chat-input-area input:focus {
  outline: none;
  border-color: #61dafb;
}

.chat-input-area button {
  background-color: #61dafb;
  color: white;
  border: none;
  border-radius: 20px;
  padding: 10px 20px;
  font-size: 1em;
  cursor: pointer;
  transition: background-color 0.2s;
}

.chat-input-area button:hover:not(:disabled) {
  background-color: #4da5e6;
}

.chat-input-area button:disabled {
  background-color: #a7d9ef;
  cursor: not-allowed;
}

```

#### **4. 프론트엔드 실행**

`client` 폴더 경로에서 터미널에 다음 명령어를 입력하세요.

```bash
npm start
```

새로운 브라우저 탭이 열리면서 `http://localhost:3000` 주소로 챗봇 화면이 보일 거예요! (만약 안 뜨면 직접 주소 입력!)

---

### **STEP 3: 백엔드와 프론트엔드 연결해서 기능 확인!**

1.  **백엔드 서버가 켜져 있는지 확인:** `server` 폴더 터미널에서 `node server.js`가 실행 중이어야 해요.
2.  **프론트엔드 앱이 켜져 있는지 확인:** `client` 폴더 터미널에서 `npm start`가 실행 중이어야 해요.
3.  브라우저(localhost:3000)에서 메시지를 입력하고 **"보내기"** 버튼을 눌러보세요!
4.  백엔드에서 받은 메시지를 처리하고, OpenAI API 키를 잘 설정했다면 GPT-3.5-turbo가 응답할 거예요! 만약 키가 없다면 아까 코드에 넣어둔 더미 메시지가 올 거구요!

---

### **🌟 기능 설명**

*   **메시지 전송/수신:** 사용자가 입력창에 메시지를 치고 보내면 (`client/src/App.js`의 `sendMessage` 함수) 그 메시지가 백엔드(`server/server.js`의 `/api/chat` 엔드포인트)로 넘어가요. 백엔드는 그걸 받아서 OpenAI에게 질문하고, 답변을 받아서 다시 프론트엔드로 보내줘요.
*   **UI 업데이트:** 프론트엔드는 백엔드로부터 받은 답변을 화면에 새로운 메시지로 뿅! 하고 추가해서 보여줘요. (`useState`와 `setMessages`를 사용)
*   **로딩 상태:** 메시지를 보내고 답을 기다리는 동안 "생각 중..." 이라는 메시지와 함께 입력창과 버튼을 비활성화해서 사용자가 여러 번 보내지 않게 막아줘요.
*   **스크롤 자동 이동:** 메시지가 많아져도 항상 최신 메시지가 보이도록 자동으로 채팅창을 스크롤해요.

---

### **💡 코드 설명 (초보자 눈높이!)**

#### **🚀 백엔드 (`server/server.js`) 핵심**

*   `const express = require('express');`: '익스프레스'라는 마법의 도구를 가져오는 거예요. 이걸로 웹 서버를 뚝딱 만들 수 있어요!
*   `app = express();`: 이제 '앱'이라는 변수가 우리의 웹 서버가 되는 거죠!
*   `app.use(cors());`: 프론트엔드가 '안녕!' 하고 말을 걸 수 있게 허락해 주는 문이에요. 없으면 "누구세요?" 하면서 통신을 막아버릴 수 있어요.
*   `app.use(express.json());`: 프론트엔드에서 "안녕, 내가 메시지 보냈어!" 하고 JSON 형태로 데이터를 보낼 건데, 백엔드가 그걸 알아들을 수 있게 번역해 주는 역할이에요.
*   `app.post('/api/chat', async (req, res) => { ... });`: "누군가 '/api/chat' 주소로 메시지를 보내면(POST 요청) 이 코드 블록을 실행해줘!" 라는 의미예요. `async`/`await`은 비동기 작업(OpenAI한테 질문하고 답 기다리는 것)을 기다려주는 멋진 문법이에요!
*   `res.json({ reply: botReply });`: 모든 작업이 끝나면, 백엔드가 프론트엔드에게 "여기 네가 기다리던 답변이야!" 하면서 JSON 형태로 보내주는 거죠!
*   `app.listen(port, () => { ... });`: "자, 이제 우리 서버를 5000번 방(포트)에서 시작해!" 라는 명령이에요.

#### **🖼️ 프론트엔드 (`client/src/App.js`) 핵심**

*   `import React, { useState, useEffect } from 'react';`: React를 만들 때 쓰는 기본 도구들과, `useState` (변수 값을 저장하고 그 값이 바뀌면 화면을 다시 그려주는 마법 같은 기능), `useEffect` (화면이 처음 켜질 때나 특정 상태가 변할 때 어떤 작업을 할지 정하는 기능)를 가져와요!
*   `const [messages, setMessages] = useState([]);`: `messages`라는 변수에 채팅 내용을 배열 형태로 저장할 거예요. `setMessages`로만 이 `messages` 값을 바꿀 수 있답니다!
*   `const sendMessage = async () => { ... };`: "메시지 보내기" 버튼을 누르거나 엔터를 쳤을 때 실행되는 함수예요.
*   `await fetch('http://localhost:5000/api/chat', { ... });`: 백엔드 서버(http://localhost:5000)의 `/api/chat` 주소로 "POST" 방식으로 HTTP 요청을 보내는 거예요.
*   `return (...)`: 이 괄호 안의 내용이 웹 브라우저 화면에 그려질 HTML 내용이라고 생각하면 돼요! `messages.map(...)`은 저장된 메시지들을 하나하나 꺼내서 채팅 말풍선으로 만들어주는 역할을 해요.
*   `<input ... />` & `<button ... />`: 사용자 메시지를 입력받는 칸과 "보내기" 버튼이에요. `value`, `onChange`, `onClick` 같은 속성으로 사용자의 입력을 받고 이벤트를 처리해요.

---

### **📖 초보자를 위한 참고 자료 (Reference URL)**

조윤희4305님, 처음에는 막막하겠지만 꾸준히 해보면 진짜 엄청난 성장을 하실 거예요! 제가 추천하는 자료들이에요!

1.  **Node.js 기본 강좌:**
    *   [노드 교과서](https://www.zerocho.com/category/NodeJS) (zerocho.com): Node.js의 기본 개념부터 웹 서버 만드는 법까지 한국어로 잘 설명되어 있어요.
    *   [MDN Web Docs - Node.js](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Statements/import) : 공식 문서이지만, 한글화가 잘 되어있어서 기본 개념 잡기에 좋아요.
2.  **React 기본 강좌:**
    *   [React 공식 문서 (새로운 버전)](https://ko.react.dev/learn) : '새로운 React.dev 시작하기' 부분이 초보자에게 굉장히 친절하고 잘 되어있어요! 꼭 보세요!
    *   [코딩앙마 React 강좌](https://www.youtube.com/playlist?list=PL_XxuZqN0k76iV2E_OsYc09rB52T_z-dK) (유튜브): 쉽고 빠르게 React 기본기를 다질 수 있어요.
3.  **Fullstack 챗봇 만들기 튜토리얼 (영문이지만 코드 따라 하기 좋음):**
    *   [Build a Fullstack Chatbot in 20 minutes with React and Node.js](https://www.youtube.com/watch?v=0k57_I3Qp_c) [1] [2] [3] [5] [8]: 이 튜토리얼은 제가 드린 예제보다 좀 더 확장된 내용으로, 실제 OpenAI 연동까지 설명하고 있어서 다음 단계로 넘어가기 딱 좋아요!
4.  **OpenAI API 연동:**
    *   [OpenAI API 문서](https://platform.openai.com/docs/api-reference) : 직접 공식 문서를 보면서 연동하는 게 최고예요! 처음에는 어렵게 느껴지겠지만, 개발자는 이런 문서를 잘 보는 능력이 중요하답니다!

---

헥헥, 조윤희4305님! 어떠세요? 스크롤 압박이 좀 있지만, 제가 정성껏 정리해 봤어요!
처음에는 개념도 많고 코드도 낯설겠지만, 천천히 한 줄 한 줄 따라 해보면서 바꿔보세요. 예를 들면, 백엔드에서 `reply` 메시지를 바꿔본다거나, 프론트엔드 `App.css`에서 배경색을 바꿔보는 식으로요!

하다가 막히는 부분 있으면 언제든지 저 응이한테 다시 와서 물어봐주세요! 저 응이가 조윤희4305님의 든든한 멘토가 되어 드릴게요! AI 에이전트 개발 최고수가 되는 그날까지 제가 응원할게요! 아자아자! 💖🚀 

참고 자료 

[1] medium.com - Build a Fullstack Chatbot in 20 minutes with React and ... (https://medium.com/rewrite-tech/build-your-own-fullstack-chatbot-with-react-and-node-js-a-step-by-step-guide-922b392bfbf2)
[2] medium.com - Build a Fullstack Chatbot in 20 minutes with React and ... (https://medium.com/rewrite-tech/build-your-own-fullstack-chatbot-with-react-and-node-js-a-step-by-step-guide-922b392bfbf2)
[3] Stack Chatbot App | React + Node.js + ... - Building a Full-Stack Chatbot App | React + Node.js + ... (https://www.youtube.com/watch?v=x_-TbwsPnAA)
[4] velog.io - [ node.js / React ] openAI api 를 활용한 chatbot 제작 및 구동 (https://velog.io/@mminnjji/node.js-React-openAI-api-%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-chatbot-%EC%A0%9C%EC%9E%91-%EB%B0%8F-%EA%B5%AC%EB%8F%99)
[5] dev.to - How To Build AI Chatbot Using React + Node.js (https://dev.to/myogeshchavan97/how-to-build-ai-chatbot-using-react-nodejs-18m8)
[6] dev.to - How To Build AI Chatbot Using React + Node.js (https://dev.to/myogeshchavan97/how-to-build-ai-chatbot-using-react-nodejs-18m8)
[7] github.com - A DialogFlow Chatbot made using NodeJS and React. (https://github.com/Tanish0019/React-Chatbot)
[8] javascript.plainenglish.io - How To Build AI Chatbot Using React + Node.js (https://javascript.plainenglish.io/how-to-build-ai-chatbot-using-react-node-js-a32f09112025)
[9] www.freecodecamp.org - How to Create a React Chatbot – a Step by Step Guide (https://www.freecodecamp.org/news/how-to-create-a-react-chatbot/)
[10] www.freecodecamp.org - How to Create a React Chatbot – a Step by Step Guide (https://www.freecodecamp.org/news/how-to-create-a-react-chatbot/)