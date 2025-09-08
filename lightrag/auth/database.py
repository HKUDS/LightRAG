"""
数据库管理模块 - 用户认证系统的数据库操作
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class UserDatabase:
    """用户数据库管理类"""
    
    def __init__(self, db_path: str = None):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径，默认为 rag_storage/users.db
        """
        if db_path is None:
            # 使用与LightRAG相同的存储目录
            storage_dir = Path("rag_storage")
            storage_dir.mkdir(exist_ok=True)
            db_path = storage_dir / "users.db"
        
        self.db_path = str(db_path)
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    full_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_superuser BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email ON users(email)")
            
            conn.commit()
            logger.info(f"数据库初始化完成: {self.db_path}")
    
    def create_user(self, user_data: Dict[str, Any]) -> bool:
        """
        创建新用户
        
        Args:
            user_data: 用户数据字典
            
        Returns:
            bool: 创建成功返回True，失败返回False
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (id, username, email, hashed_password, full_name, is_active, is_superuser)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_data['id'],
                    user_data['username'],
                    user_data['email'],
                    user_data['hashed_password'],
                    user_data.get('full_name'),
                    user_data.get('is_active', True),
                    user_data.get('is_superuser', False)
                ))
                conn.commit()
                logger.info(f"用户创建成功: {user_data['username']}")
                return True
        except sqlite3.IntegrityError as e:
            logger.error(f"用户创建失败，用户名或邮箱已存在: {e}")
            return False
        except Exception as e:
            logger.error(f"用户创建失败: {e}")
            return False
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        根据用户名获取用户信息
        
        Args:
            username: 用户名
            
        Returns:
            Dict[str, Any]: 用户信息字典，不存在返回None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        根据用户ID获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 用户信息字典，不存在返回None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        根据邮箱获取用户信息
        
        Args:
            email: 邮箱地址
            
        Returns:
            Dict[str, Any]: 用户信息字典，不存在返回None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            update_data: 要更新的数据字典
            
        Returns:
            bool: 更新成功返回True，失败返回False
        """
        try:
            # 构建动态更新语句
            update_fields = []
            values = []
            
            for field, value in update_data.items():
                if field != 'id':  # 不允许更新ID
                    update_fields.append(f"{field} = ?")
                    values.append(value)
            
            if not update_fields:
                return True
            
            # 添加更新时间
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(user_id)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                sql = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(sql, values)
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"用户更新成功: {user_id}")
                    return True
                else:
                    logger.warning(f"用户不存在: {user_id}")
                    return False
        except Exception as e:
            logger.error(f"用户更新失败: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: 删除成功返回True，失败返回False
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"用户删除成功: {user_id}")
                    return True
                else:
                    logger.warning(f"用户不存在: {user_id}")
                    return False
        except Exception as e:
            logger.error(f"用户删除失败: {e}")
            return False
    
    def list_users(self, limit: int = 100, offset: int = 0) -> list:
        """
        获取用户列表
        
        Args:
            limit: 限制返回数量
            offset: 偏移量
            
        Returns:
            list: 用户列表
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, email, full_name, is_active, is_superuser, created_at, updated_at
                    FROM users 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"获取用户列表失败: {e}")
            return []
    
    def count_users(self) -> int:
        """
        获取用户总数
        
        Returns:
            int: 用户总数
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"获取用户总数失败: {e}")
            return 0


# 全局数据库实例
_db_instance = None

def get_user_db() -> UserDatabase:
    """获取用户数据库实例（单例模式）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = UserDatabase()
    return _db_instance
