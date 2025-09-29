# backend/app/api/auth.py
from fastapi import APIRouter, HTTPException, status, Request, Response
from fastapi.responses import RedirectResponse
from app.core.config import settings
from app.core.security import create_access_token
import httpx  # Google OAuth2 API 호출을 위해 필요
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/google/login")
async def google_login():
    """
    Google OAuth2 로그인 페이지로 리다이렉트합니다.
    """
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/auth"
        f"?response_type=code"
        f"&client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
        f"&scope=openid%20email%20profile"  # 이메일, 프로필 정보 요청
        f"&access_type=offline"  # Refresh token을 받기 위해
    )
    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_callback(request: Request):
    """
    Google OAuth2 콜백을 처리하고 JWT 토큰을 발행합니다.
    """
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="인증 코드를 받지 못했습니다.")

    # 1. Access Token 요청
    token_url = "http://localhost:3000/auth?access_token"
    #token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=token_data)
        token_response.raise_for_status()
        token_json = token_response.json()

    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Google Access Token 발행 실패")

    # 2. User Info 요청
    userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(userinfo_url, headers=headers)
        userinfo_response.raise_for_status()
        userinfo_json = userinfo_response.json()

    user_email = userinfo_json.get("email")
    user_name = userinfo_json.get("name")
    if not user_email:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="사용자 이메일 정보를 가져올 수 없습니다.")

    # 3. JWT Access Token 생성 및 반환
    user_data_for_jwt = {"email": user_email, "name": user_name, "id": userinfo_json.get("id")}
    jwt_access_token = create_access_token(user_data_for_jwt)

    # 실제 챗봇 UI로 리다이렉트 (프론트엔드 URL에 토큰을 포함하거나 쿠키로 설정)
    # 여기서는 간단히 JSON 응답으로 토큰을 반환합니다.
    # 실제 앱에서는 클라이언트(챗봇 UI)가 이 토큰을 받아서 다음 API 호출 시 Header에 Bearer Token으로 사용해야 합니다.
    return {"access_token": jwt_access_token, "token_type": "bearer", "user": user_data_for_jwt}
