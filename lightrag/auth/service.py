from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4
import bcrypt
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError

from .models.user import UserInDB, UserCreate, UserResponse
from .database import get_user_db

# Configuration - move to settings in production
SECRET_KEY = "your-secret-key-here"  # Change this to a strong secret key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Password hashing
def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Authentication service
class AuthService:
    @staticmethod
    def create_user(user_data: UserCreate) -> UserInDB:
        # Check if user already exists
        db = get_user_db()
        if db.get_user_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        if db.get_user_by_username(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Create new user
        user_dict = user_data.dict()
        hashed_password = get_password_hash(user_dict.pop("password"))
        user_id = str(uuid4())
        
        user_db_data = {
            'id': user_id,
            'username': user_dict['username'],
            'email': user_dict['email'],
            'hashed_password': hashed_password,
            'full_name': user_dict.get('full_name'),
            'is_active': True,
            'is_superuser': False
        }
        
        # Store user in database
        if not db.create_user(user_db_data):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Return UserInDB object
        return UserInDB(
            id=UUID(user_id),
            username=user_dict['username'],
            email=user_dict['email'],
            hashed_password=hashed_password,
            full_name=user_dict.get('full_name'),
            is_active=True,
            is_superuser=False
        )
    
    @staticmethod
    def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
        # Find user by email
        db = get_user_db()
        user_data = db.get_user_by_email(email)
        if not user_data:
            return None
        if not verify_password(password, user_data['hashed_password']):
            return None
        
        # Convert to UserInDB object
        return UserInDB(
            id=UUID(user_data['id']),
            username=user_data['username'],
            email=user_data['email'],
            hashed_password=user_data['hashed_password'],
            full_name=user_data.get('full_name'),
            is_active=user_data['is_active'],
            is_superuser=user_data['is_superuser']
        )
    
    @staticmethod
    async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = decode_token(token)
            user_id = payload.get("sub")
            if user_id is None:
                raise credentials_exception
        except (InvalidTokenError, ValidationError):
            raise credentials_exception
        
        db = get_user_db()
        user_data = db.get_user_by_id(user_id)
        if user_data is None:
            raise credentials_exception
        
        # Convert to UserInDB object
        return UserInDB(
            id=UUID(user_data['id']),
            username=user_data['username'],
            email=user_data['email'],
            hashed_password=user_data['hashed_password'],
            full_name=user_data.get('full_name'),
            is_active=user_data['is_active'],
            is_superuser=user_data['is_superuser']
        )
    
    @staticmethod
    async def login_for_access_token(username: str, password: str) -> dict:
        user = AuthService.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
