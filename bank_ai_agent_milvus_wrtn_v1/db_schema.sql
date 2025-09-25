-- db_schema.sql

-- 사용자 세션 테이블
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    messages JSONB DEFAULT '[]'::jsonb, -- 채팅 히스토리 (JSONB 타입)
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 추가 (조회 성능 향상)
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_email ON chat_sessions(user_email);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity ON chat_sessions(last_activity);


-- 사용자 피드백 테이블
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE SET NULL, -- 세션 삭제 시 NULL로 설정
    message_id VARCHAR(255) NOT NULL, -- 특정 응답 메시지 ID (프론트엔드에서 생성)
    feedback_type VARCHAR(50) NOT NULL, -- 'like', 'dislike'
    user_query TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_feedback_user_email ON feedback(user_email);
CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback(session_id);
