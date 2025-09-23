"""Authentication utilities with JWT token handling.
- create_access_token: create a JWT with expiration
- get_current_user: dependency that decodes JWT and returns user info

This is a demo implementation. For production:
- Use rotating signing keys, key IDs (kid), and proper secret management (HashiCorp Vault)
- Consider OIDC or enterprise SSO integration
"""
import os
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '60'))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

# Demo user store (replace with DB in production)
_demo_users = {'alice': {'username':'alice', 'hashed_password':pwd_context.hash('password'), 'scopes':['read','write'] }}


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def authenticate_user(username, password):
    u = _demo_users.get(username)
    if not u: return False
    return verify_password(password, u['hashed_password'])


def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode.update({'exp': expire})
    encoded = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate credentials')
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise credentials_exception
        return {'username': username}
    except JWTError:
        raise credentials_exception
