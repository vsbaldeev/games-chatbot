"""v3 initial schema.

Baseline schema for the v3 application. Captures the complete schema previously
created at bot startup by the various ``init_table``/``init_tables`` helpers.
DDL is emitted as raw SQL because the application talks to PostgreSQL through
asyncpg with no SQLAlchemy models.

For a database that already contains these tables (created by an earlier
bot version), baseline it instead of running this migration::

    alembic stamp head

Revision ID: v3_0001
Revises:
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE chat_members (
            chat_id  BIGINT   NOT NULL,
            user_id  BIGINT   NOT NULL,
            username TEXT,
            is_bot   BOOLEAN  NOT NULL DEFAULT FALSE,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    op.execute("""
        CREATE TABLE user_stats (
            user_id              BIGINT  NOT NULL,
            chat_id              BIGINT  NOT NULL,
            username             TEXT,
            laugh_reactions      INTEGER NOT NULL DEFAULT 0,
            heart_reactions      INTEGER NOT NULL DEFAULT 0,
            fire_reactions       INTEGER NOT NULL DEFAULT 0,
            thumbsup_reactions   INTEGER NOT NULL DEFAULT 0,
            emoji_messages       INTEGER NOT NULL DEFAULT 0,
            sticker_messages     INTEGER NOT NULL DEFAULT 0,
            forwarded_messages   INTEGER NOT NULL DEFAULT 0,
            link_messages        INTEGER NOT NULL DEFAULT 0,
            voice_messages       INTEGER NOT NULL DEFAULT 0,
            video_messages       INTEGER NOT NULL DEFAULT 0,
            video_note_messages  INTEGER NOT NULL DEFAULT 0,
            photo_messages       INTEGER NOT NULL DEFAULT 0,
            night_messages       INTEGER NOT NULL DEFAULT 0,
            animation_messages   INTEGER NOT NULL DEFAULT 0,
            roasted_count        INTEGER NOT NULL DEFAULT 0,
            roulette_win_count   INTEGER NOT NULL DEFAULT 0,
            duel_wins            INTEGER NOT NULL DEFAULT 0,
            long_messages        INTEGER NOT NULL DEFAULT 0,
            voice_max_duration   INTEGER NOT NULL DEFAULT 0,
            long_message_max     INTEGER NOT NULL DEFAULT 0,
            last_seen            BIGINT  NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, chat_id)
        )
    """)
    op.execute("""
        CREATE TABLE announced_achievements (
            user_id BIGINT NOT NULL,
            chat_id BIGINT NOT NULL,
            key     TEXT   NOT NULL,
            PRIMARY KEY (user_id, chat_id, key)
        )
    """)
    op.execute("""
        CREATE TABLE message_authors (
            chat_id    BIGINT NOT NULL,
            message_id BIGINT NOT NULL,
            user_id    BIGINT NOT NULL,
            username   TEXT   NOT NULL,
            created_at BIGINT NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    op.execute("""
        CREATE TABLE message_reaction_counts (
            chat_id     BIGINT  NOT NULL,
            message_id  BIGINT  NOT NULL,
            emoji       TEXT    NOT NULL,
            total_count INTEGER NOT NULL,
            updated_at  BIGINT  NOT NULL,
            PRIMARY KEY (chat_id, message_id, emoji)
        )
    """)

    op.execute("""
        CREATE TABLE unified_messages (
            message_id      BIGINT           NOT NULL,
            chat_id         BIGINT           NOT NULL,
            user_id         BIGINT           NOT NULL,
            username        TEXT             NOT NULL,
            content         TEXT             NOT NULL,
            media_type      TEXT             NOT NULL DEFAULT 'text',
            reply_to_msg_id BIGINT,
            file_id         TEXT,
            created_at      DOUBLE PRECISION NOT NULL,
            media_group_id  TEXT,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_unified_messages_chat_time
        ON unified_messages (chat_id, created_at DESC)
    """)
    op.execute("""
        CREATE INDEX idx_unified_messages_media_group
        ON unified_messages (chat_id, media_group_id)
        WHERE media_group_id IS NOT NULL
    """)

    op.execute("""
        CREATE TABLE user_memories (
            id         BIGSERIAL        PRIMARY KEY,
            chat_id    BIGINT           NOT NULL,
            user_id    BIGINT           NOT NULL,
            username   TEXT             NOT NULL,
            fact       TEXT             NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            embedding  vector(384)
        )
    """)
    op.execute("""
        CREATE INDEX idx_user_memories_lookup
        ON user_memories (chat_id, user_id)
    """)
    op.execute("""
        CREATE INDEX idx_user_memories_hnsw
        ON user_memories USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
    """)

    op.execute("""
        CREATE TABLE thread_history (
            thread_id  TEXT             NOT NULL,
            chat_id    BIGINT           NOT NULL,
            role       TEXT             NOT NULL,
            content    TEXT             NOT NULL,
            created_at DOUBLE PRECISION NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX idx_thread_history_lookup
        ON thread_history (thread_id, created_at)
    """)

    op.execute("""
        CREATE TABLE roast_log (
            message_id     BIGINT           NOT NULL,
            chat_id        BIGINT           NOT NULL,
            target_user_id BIGINT           NOT NULL,
            anchor_key     TEXT             NOT NULL,
            created_at     DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (message_id, chat_id)
        )
    """)
    op.execute("""
        CREATE TABLE roast_reactions (
            message_id  BIGINT  NOT NULL,
            chat_id     BIGINT  NOT NULL,
            emoji       TEXT    NOT NULL,
            count       INT     NOT NULL DEFAULT 0,
            PRIMARY KEY (message_id, chat_id, emoji)
        )
    """)
    op.execute("""
        CREATE TABLE roast_queue (
            chat_id BIGINT NOT NULL,
            user_id BIGINT NOT NULL,
            PRIMARY KEY (chat_id, user_id)
        )
    """)

    op.execute("""
        CREATE TABLE user_tags (
            chat_id     BIGINT           NOT NULL,
            user_id     BIGINT           NOT NULL,
            tag         TEXT             NOT NULL,
            reason      TEXT             NOT NULL,
            assigned_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (chat_id, user_id)
        )
    """)

    op.execute("""
        CREATE TABLE sent_memes (
            chat_id BIGINT NOT NULL,
            url     TEXT   NOT NULL,
            PRIMARY KEY (chat_id, url)
        )
    """)


def downgrade() -> None:
    raise NotImplementedError("This service does not support schema downgrades.")
