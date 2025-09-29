// server/server.js
const express = require('express');
const cors = require('cors');
const axios = require('axios'); // axios 라이브러리 불러오기 (OpenAI 대신 사용할 거예요)
// const { OpenAI } = require('openai'); // ❌ OpenAI 라이브러리는 이제 필요 없어요! 주석 처리하거나 지워주세요.
// require('dotenv').config(); // ❌ .env 파일도 이제 필요 없어요! 주석 처리하거나 지워주세요.

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

// ❌ OpenAI API 키 설정 관련 부분도 이제 필요 없어요! 주석 처리하거나 지워주세요.
// const openai = new OpenAI({
//     apiKey: process.env.OPENAI_API_KEY,
// });

// 테스트용 루트 경로 API (그대로 두셔도 돼요)
app.get('/', (req, res) => {
    res.send('Chatbot Backend is running with Ollama!');
});

// 챗봇과의 대화를 처리할 API 엔드포인트
app.post('/api/chat', async (req, res) => {
    const userMessage = req.body.message;
    console.log(`Received message: ${userMessage}`);

    try {
        // ✨ Ollama 로컬 API 호출!
        const ollamaResponse = await axios.post('http://localhost:11434/api/chat', { // Ollama API 기본 주소
            model: 'llama2', // ✨ 여기를 조윤희4305님이 설치한 모델 이름으로 변경해주세요! (예: 'phi3')
            messages: [{ role: 'user', content: userMessage }],
            stream: false, // 실시간 스트리밍 대신 최종 답변을 한 번에 받을게요.
        }, {
            headers: {
                'Content-Type': 'application/json',
            },
        });

        const botReply = ollamaResponse.data.message.content; // Ollama 응답에서 답변 내용 추출
        console.log(`Bot reply from Ollama: ${botReply}`);
        res.json({ reply: botReply }); // AI의 답변을 프론트엔드로 보내줘요.

    } catch (error) {
        console.error("Error calling Ollama API:", error.message);
        // Ollama 서버가 실행 중이 아니거나 모델을 찾을 수 없을 때의 에러 처리
        if (error.code === 'ECONNREFUSED' || error.response?.status === 404) {
            res.status(503).json({ reply: "죄송해요, 로컬 AI 모델(Ollama)에 연결할 수 없거나 모델을 찾을 수 없어요. Ollama 서버를 실행하고 'llama2' 모델(또는 설정한 모델)이 준비되었는지 확인해주세요. ㅠㅠ" });
        } else {
            res.status(500).json({ reply: `죄송해요, AI와 대화 중에 오류가 발생했어요. ${error.message}` });
        }
    }
});

// 서버 시작!
app.listen(port, () => {
    console.log(`Backend server listening at http://localhost:${port}`);
    console.log('Ctrl + C 를 눌러 서버를 종료할 수 있습니다.');
});

