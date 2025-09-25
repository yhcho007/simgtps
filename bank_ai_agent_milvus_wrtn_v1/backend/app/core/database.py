# backend/app/core/database.py
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from app.core.config import settings
import logging
import asyncio # for asyncio.to_thread

logger = logging.getLogger(__name__)

# 전역 DB 연결 풀
db_pool: Optional[SimpleConnectionPool] = None

async def connect_to_db():
    global db_pool
    if db_pool is None:
        try:
            db_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10, # 필요에 따라 커넥션 풀 크기 조절
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DB
            )
            # 연결 테스트
            conn = await asyncio.to_thread(db_pool.getconn)
            cursor = await asyncio.to_thread(conn.cursor)
            await asyncio.to_thread(cursor.execute, "SELECT 1")
            await asyncio.to_thread(cursor.close)
            await asyncio.to_thread(db_pool.putconn, conn)
            logger.info("PostgreSQL 데이터베이스에 성공적으로 연결되었습니다.")
        except Exception as e:
            logger.error(f"PostgreSQL 데이터베이스 연결 실패: {e}", exc_info=True)
            raise RuntimeError(f"PostgreSQL 연결 실패: {e}")

async def close_db_connection():
    global db_pool
    if db_pool:
        logger.info("PostgreSQL 데이터베이스 연결을 종료합니다.")
        await asyncio.to_thread(db_pool.closeall)
        db_pool = None

async def get_db_connection():
    if db_pool is None:
        raise RuntimeError("데이터베이스 연결 풀이 초기화되지 않았습니다.")
    # Thread pool executor를 사용하여 동기 psycopg2 호출을 비동기로 실행
    return await asyncio.to_thread(db_pool.getconn)

async def release_db_connection(conn):
    if db_pool and conn:
        await asyncio.to_thread(db_pool.putconn, conn)

async def execute_query(query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = False):
    conn = None
    try:
        conn = await get_db_connection()
        # NamedTupleCursor를 사용하여 결과가 컬럼 이름으로 접근 가능한 객체로 반환되도록 함
        cursor = await asyncio.to_thread(conn.cursor, cursor_factory=psycopg2.extras.NamedTupleCursor)
        await asyncio.to_thread(cursor.execute, query, params)
        if fetch_one:
            result = await asyncio.to_thread(cursor.fetchone)
        elif fetch_all:
            result = await asyncio.to_thread(cursor.fetchall)
        else:
            result = None
        await asyncio.to_thread(conn.commit) # DDL/DML 후에 커밋
        await asyncio.to_thread(cursor.close)
        return result
    except Exception as e:
        if conn:
            await asyncio.to_thread(conn.rollback) # 오류 발생 시 롤백
        logger.error(f"DB 쿼리 실행 중 오류 발생: {e}, 쿼리: {query}", exc_info=True)
        raise RuntimeError(f"데이터베이스 오류: {e}")
    finally:
        if conn:
            await release_db_connection(conn)
