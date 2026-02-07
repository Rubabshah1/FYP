"""
Supabase client initialization and configuration.
"""
import os
from supabase import create_client, Client
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system environment variables

# Supabase configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY"))

# Initialize Supabase client
supabase: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get or create Supabase client instance."""
    global supabase
    if supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment variables. "
                "Get these from your Supabase project settings:\n"
                "1. Go to your Supabase project dashboard\n"
                "2. Navigate to Settings → API\n"
                "3. Copy the Project URL and anon/public key\n"
                "4. Set them as environment variables:\n"
                "   export SUPABASE_URL='https://xxxxx.supabase.co'\n"
                "   export SUPABASE_KEY='your-anon-key-here'"
            )
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

def init_supabase():
    """Initialize Supabase client. Returns True if successful, False otherwise."""
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("  WARNING: SUPABASE_URL and SUPABASE_KEY are not set.")
            print("   The application will fail when trying to access the database.")
            print("   Please set these environment variables:")
            print("   export SUPABASE_URL='https://xxxxx.supabase.co'")
            print("   export SUPABASE_KEY='your-anon-key-here'")
            print("   Or create a .env file with these variables.")
            return False
        get_supabase_client()
        print(" Supabase client initialized successfully")
        return True
    except Exception as e:
        print(f" Failed to initialize Supabase client: {e}")
        print("   Please check your Supabase credentials and try again.")
        return False

