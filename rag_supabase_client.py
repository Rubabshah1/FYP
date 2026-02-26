from typing import Optional

from supabase import Client, create_client

from rag_config import SUPABASE_URL, SUPABASE_KEY

_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client (singleton)."""
    global _SUPABASE_CLIENT

    if _SUPABASE_CLIENT is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file\n"
                "Get these from: Supabase Dashboard -> Settings -> API"
            )

        print(f"Connecting to Supabase: {SUPABASE_URL}")
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)

    return _SUPABASE_CLIENT


def test_connection() -> bool:
    """Test Supabase connection and pgvector availability."""
    try:
        supabase = get_supabase_client()
        # Simple table probe
        _ = (
            supabase.table("documents")
            .select("doc_id")
            .limit(1)
            .execute()
        )

        print("✅ Connected to Supabase successfully!")
        print(f" URL: {SUPABASE_URL}")

        # Check pgvector function
        try:
            _ = supabase.rpc(
                "match_documents_simple",
                {"query_embedding": [0.0] * 768, "match_count": 1},
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

