from api.db import SessionLocal  # noqa: E402
from pipeline.topics.topics import merge_aliases  # noqa: E402

session = SessionLocal()
updated_count = merge_aliases(session)
print(f"Updated {updated_count} entities with alias_of")
session.close()