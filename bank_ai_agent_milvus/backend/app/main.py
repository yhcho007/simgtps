"""FastAPI main application with:
- JWT auth endpoints
- WebSocket chat endpoint (token optional for demo)
- REST chat endpoint (requires token)
- Feedback endpoint to collect satisfaction scores and reasons
- Prometheus metrics endpoint

Each function has comments explaining behavior and security notes.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.services.agent_service import AgentService
from app.auth import get_current_user, create_access_token, authenticate_user
from app.models.user import UserIn, Token
from prometheus_client import make_asgi_app
import os

app = FastAPI(title="Bank AI Agent - Full Prod Demo")

# Allow local frontend origin for development. In production, tighten this list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Prometheus ASGI app and mount under /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

agent = AgentService()

@app.post('/login', response_model=Token)
def login(user: UserIn):
    """Authenticate demo user and return JWT access token.
    In production, replace authenticate_user with real user DB and MFA.
    """
    if not authenticate_user(user.username, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    access_token = create_access_token({'sub': user.username})
    return {'access_token': access_token, 'token_type': 'bearer'}

@app.post('/chat/send')
def send_message(payload: dict, current_user=Depends(get_current_user)):
    """REST endpoint for sending chat messages. Requires JWT token via Authorization header.
    Payload should include: session_id, text, optional user info (account_id)
    """
    session_id = payload.get('session_id', 'anon')
    text = payload.get('text', '')
    return agent.handle_user_message(session_id, text, dict(username=current_user['username'], **payload.get('user', {})))

@app.post('/chat/feedback')
def feedback(payload: dict, current_user=Depends(get_current_user)):
    """Collects satisfaction rating (1-10) and optional feedback reason.
    If rating is 1-3, feedback['reason'] is required and will be stored for triage.
    """
    session_id = payload.get('session_id', 'anon')
    msg_id = payload.get('msg_id')
    rating = int(payload.get('rating', 0))
    reason = payload.get('reason', '').strip()
    if rating < 1 or rating > 10:
        raise HTTPException(status_code=400, detail='rating must be between 1 and 10')
    if rating <= 3 and not reason:
        # Force user (or frontend) to provide reason on low rating for troubleshooting
        raise HTTPException(status_code=400, detail='reason required for ratings 1-3')
    # Delegate to agent service to store feedback
    agent.record_feedback(session_id=session_id, msg_id=msg_id, user={'username': current_user['username']}, rating=rating, reason=reason)
    return {'status':'ok'}

@app.websocket('/ws/chat')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket chat endpoint for realtime messaging.
    For simplicity this demo doesn't require token on WS connect, but in production
    pass token as a query param or initial message and validate it.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            session_id = data.get('session_id', 'anon')
            text = data.get('text', '')
            user = data.get('user', {'id':'demo_user'})
            resp = agent.handle_user_message(session_id, text, user)
            await websocket.send_json(resp)
    except WebSocketDisconnect:
        print('WebSocket disconnected')

@app.get('/health')
def health():
    return {'status':'ok'}
