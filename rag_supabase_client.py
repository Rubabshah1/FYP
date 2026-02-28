from typing import Optional

from supabase import Client, create_client

from rag_config import SUPABASE_URL, SUPABASE_KEY, EMBEDDING_DIM

_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase_client() -> Client:
    global _SUPABASE_CLIENT

    if _SUPABASE_CLIENT is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_KEY) must be set."
            )
        print(f"Connecting to Supabase: {SUPABASE_URL}")
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)

    return _SUPABASE_CLIENT


def test_connection() -> bool:
    try:
        supabase = get_supabase_client()
        _ = supabase.table("documents").select("doc_id").limit(1).execute()
        print("✅ Connected to Supabase successfully!")
        try:
            _ = supabase.rpc(
                "match_documents_simple",
                {"query_embedding": [0.0] * EMBEDDING_DIM, "match_count": 1},
            ).execute()
            print("✅ pgvector extension verified")
        except Exception as e:
            if "does not exist" in str(e):
                print("⚠️ match_documents function not found - run the SQL setup")
            else:
                print(f"⚠️ Could not verify pgvector: {e}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False