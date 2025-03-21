import os
import psycopg2
from psycopg2.extras import DictCursor


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "course_assistant"),
        user=os.getenv("POSTGRES_USER", "your_username"),
        password=os.getenv("POSTGRES_PASSWORD", "your_password"),
    )


def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    course TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("NEw tables created.")
        conn.commit()
        print("Database Initiazation completed Succesfully")
    except Exception as e:
        print(f"an error occured during database initialization: {e}")
        conn.rollback()
    finally:
        conn.close()


def save_conversation(conversation_id, question, answer, course):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, question, answer, course) VALUES (%s, %s, %s, %s)",
                (conversation_id, question, answer, course),
            )
        conn.commit()
        print(f"Conversation saved with id: {conversation_id}")
    except Exception as e:
        print(f"an error occured while saving the conversation: {e}")
        conn.rollback()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback) VALUES(%s, %s)",
                (conversation_id, feedback),
            )
        conn.commit()
        print(f"Feedback saved for conversation Id: {conversation_id}")
    except Exception as e:
        print(f"an error occured while saving the feedback: {e}")
        conn.rollback()
    finally:
        conn.close()



