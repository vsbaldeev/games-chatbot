"""Index bot_reply_labels on the column pair the hourly rollup joins on.

Reply reaction metrics (docs/superpowers/specs/2026-10-09-reply-reaction-metrics-design.md).

v5_0004 gave bot_reply_labels its primary key (chat_id, reply_message_id) and
two created_at indexes, and nothing covering (chat_id, bot_message_id). But
ROLLUP_SQL (src/store/feedback_metrics.py) now carries five correlated EXISTS
subqueries inside aggregate FILTER clauses, every one of them matching
`label.chat_id = feedback.chat_id AND label.bot_message_id =
feedback.message_id` — and that query runs every 300 seconds
(feedback_metrics_job, src/app/jobs.py).

Without the index Postgres hashes each subquery, which means five full scans of
bot_reply_labels per rollup. Measured on PostgreSQL 18.4 against 415k label
rows and 20k feedback rows: 434 ms per rollup, five Seq Scans hashing 62k-83k
rows each. With this index the same rollup plans as five Index Only Scans and
runs in 4.8 ms.

INCLUDE (reaction, media_type) carries the two columns those FILTER clauses
test, so the EXISTS is answered from the index alone — the measured plan is an
Index Only Scan at an estimated cost of 4.90 per probe, against 43.90 for a
plain two-column index that still has to visit the heap. INCLUDE needs
PostgreSQL 11+; docker-compose.yml pins pgvector/pgvector:pg18-trixie, and the
measurements above are from that image.

The table takes writes only twice per row (one INSERT at enqueue, one UPDATE at
write_label) against a read every 300 seconds, so the index's write cost is not
a consideration here.

Revision ID: v5_0005
Revises: v5_0004
Create Date: 2026-10-12
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0005"
down_revision: Union[str, None] = "v5_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX idx_bot_reply_labels_bot_message
            ON bot_reply_labels (chat_id, bot_message_id)
            INCLUDE (reaction, media_type)
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
