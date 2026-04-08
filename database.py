import sqlite3
import asyncio
from datetime import datetime
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DB_FILE = "chat_history.db"

def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        
        # Create threads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(thread_id) REFERENCES threads(id)
            )
        """)
        conn.commit()

init_db()

# Sync helpers that we will run via asyncio.to_thread

def _create_user(username, password):
    hash_password = pwd_context.hash(password)
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password))
            conn.commit()
            return {"id": cursor.lastrowid, "username": username}
        except sqlite3.IntegrityError:
            return None # Username exists

def _verify_user(username, password):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and pwd_context.verify(password, user[2]):
            return {"id": user[0], "username": user[1]}
        return None

def _create_thread(user_id, title="New Chat"):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO threads (user_id, title) VALUES (?, ?)", (user_id, title))
        conn.commit()
        return {"id": cursor.lastrowid, "title": title}

def _get_threads(user_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at FROM threads WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        return [{"id": row[0], "title": row[1], "created_at": row[2]} for row in cursor.fetchall()]

def _add_message(thread_id, role, content):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)", (thread_id, role, content))
        conn.commit()
        
        # Auto-update thread title if it's the first user message
        if role == 'user':
            cursor.execute("SELECT id FROM messages WHERE thread_id = ?", (thread_id,))
            msgs = cursor.fetchall()
            if len(msgs) == 1: # First message
                title_preview = content[:30] + '...' if len(content) > 30 else content
                cursor.execute("UPDATE threads SET title = ? WHERE id = ?", (title_preview, thread_id))
                conn.commit()

def _get_messages(thread_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM messages WHERE thread_id = ? ORDER BY created_at ASC", (thread_id,))
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]


# Async Interfaces

async def create_user_async(username, password):
    return await asyncio.to_thread(_create_user, username, password)

async def verify_user_async(username, password):
    return await asyncio.to_thread(_verify_user, username, password)

async def create_thread_async(user_id, title="New Chat"):
    return await asyncio.to_thread(_create_thread, user_id, title)

async def get_threads_async(user_id):
    return await asyncio.to_thread(_get_threads, user_id)

async def add_message_async(thread_id, role, content):
    await asyncio.to_thread(_add_message, thread_id, role, content)

async def get_messages_async(thread_id):
    return await asyncio.to_thread(_get_messages, thread_id)
