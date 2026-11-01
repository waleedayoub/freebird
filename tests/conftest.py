import os
import tempfile

# config.py reads required settings at import time; give it dummies and a throwaway data dir.
os.environ.setdefault("VICOHOME_EMAIL", "test@example.com")
os.environ.setdefault("VICOHOME_PASSWORD", "x")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:test")
os.environ.setdefault("TELEGRAM_CHAT_ID", "1")
os.environ["FREEBIRD_DATA_DIR"] = tempfile.mkdtemp(prefix="freebird-test-")
