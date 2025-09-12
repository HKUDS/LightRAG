from sqlalchemy import create_engine, NullPool, text
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DATABASE = os.getenv("DB_DATABASE")

url = URL.create(
    "mariadb+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_DATABASE,
)

engine = create_engine(
    url,
    connect_args={
        "ssl_verify_identity": True,
    },
    pool_pre_ping=True,
    poolclass=NullPool,
)

with engine.begin() as conn:
    print(conn.execute(text("SELECT 1")).scalar())

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

schema_sql = r"""
-- your script (USE is optional since we're connected to the DB already)
CREATE TABLE IF NOT EXISTS edumind.users (
  id               VARCHAR(36) PRIMARY KEY,
  username         VARCHAR(255)   NOT NULL UNIQUE,
  password         VARCHAR(255)   NULL,
  email            VARCHAR(255)   NOT NULL UNIQUE,
  full_name        VARCHAR(255)   NULL,
  created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edumind.teams (
  id               VARCHAR(36) PRIMARY KEY,
  name             VARCHAR(255)   NOT NULL UNIQUE,
  description      TEXT           NULL,
  created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edumind.user_teams (
  user_id          VARCHAR(36) NOT NULL,
  team_id          VARCHAR(36) NOT NULL,
  PRIMARY KEY (user_id, team_id),
  created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edumind.projects(
  id                    VARCHAR(36) PRIMARY KEY,
  name                  VARCHAR(255),
  instructions          TEXT,
  user_id               VARCHAR(36) NOT NULL,
  created_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edumind.files (
  id                    VARCHAR(36) PRIMARY KEY,
  user_id               VARCHAR(36) NOT NULL,
  project_id            VARCHAR(36) NOT NULL,
  filename              VARCHAR(512)   NOT NULL,
  file_type             ENUM('PDF','DOCX','TXT','PPT') NOT NULL,
  size                  BIGINT         NOT NULL,
  file_path             VARCHAR(1024)  NOT NULL,
  uploaded_at           DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  text_extracted_flag   TINYINT(1)     NOT NULL DEFAULT 0,
  extracted_text_path   VARCHAR(1024)  NULL,
  file_scan_flag        TINYINT(1)     NOT NULL DEFAULT 0,
  error_message         TEXT           NULL
);

CREATE TABLE IF NOT EXISTS edumind.chat_sessions (
  id               VARCHAR(36)    PRIMARY KEY,
  topic            VARCHAR(255)   NULL,
  memory_state     JSON           NULL,
  user_id          VARCHAR(36)    NOT NULL,
  project_id       VARCHAR(36)    NOT NULL,
  created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_active_at   DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edumind.chat_messages (
  id               VARCHAR(36) PRIMARY KEY,
  session_id       VARCHAR(36) NOT NULL,
  role             ENUM('user','assistant') NOT NULL,
  content          TEXT           NOT NULL,
  output           TEXT           NULL,
  created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_cm_session (session_id)
);

CREATE TABLE IF NOT EXISTS edumind.questions (
  id               VARCHAR(36) PRIMARY KEY,
  user_id          VARCHAR(36) NOT NULL,
  session_id       VARCHAR(36) NOT NULL,
  project_id       VARCHAR(36) NULL,
  question_text    TEXT           NOT NULL,
  options          JSON           NOT NULL,
  correct_answers  JSON           NOT NULL,
  difficulty_level VARCHAR(100)   NULL, 
  tags             JSON           NULL,
  source           TEXT           NOT NULL,
  isApproved       TINYINT DEFAULT 0,
  isArchived       TINYINT DEFAULT 0,
  created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT IGNORE INTO edumind.users (id, username, password, email, full_name)
VALUES ('101', 'testuser', 'hashedpassword123', 'testuser@example.com', 'Test User');

INSERT IGNORE INTO edumind.teams (id, name, description)
VALUES ('201', 'Test Team', 'This is a test team');

INSERT IGNORE INTO edumind.user_teams (user_id, team_id)
VALUES ('101', '201');
"""

stmts = [s.strip() for s in schema_sql.split(";") if s.strip()]
with engine.begin() as conn:
    # CREATE SCHEMA if not done above (harmless if already exists)
    conn.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS `{DB_DATABASE}`;")
    # Apply the rest
    for s in stmts:
        conn.exec_driver_sql(s)

print("✅ Database and tables are ready.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()