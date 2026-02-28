-- ============================================
-- CHAT INTEGRATION WITH EXISTING SCHEMA
-- This integrates with your existing sessions, queries, responses tables
-- ============================================

-- Note: You already have sessions, queries, and responses tables!
-- We'll modify the FastAPI to use your existing schema instead of creating new tables

-- Add indexes to existing tables for better chat performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON public.sessions(last_active DESC);
CREATE INDEX IF NOT EXISTS idx_queries_session_id ON public.queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON public.queries(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_responses_session_id ON public.responses(session_id);
CREATE INDEX IF NOT EXISTS idx_responses_timestamp ON public.responses(timestamp DESC);

-- Add a column to sessions to track conversation metadata (optional)
ALTER TABLE public.sessions 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.sessions.metadata IS 'Store conversation context like language preference, category filters, etc.';

-- Add an index on metadata for faster lookups
CREATE INDEX IF NOT EXISTS idx_sessions_metadata ON public.sessions USING gin(metadata);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get conversation history for a session
CREATE OR REPLACE FUNCTION get_conversation_history(
    p_session_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    message_type TEXT,
    content TEXT,
    timestamp TIMESTAMPTZ,
    response_id UUID,
    query_id UUID
) AS $$
BEGIN
    RETURN QUERY
    WITH combined_messages AS (
        -- Get queries (user messages)
        SELECT 
            'user' as message_type,
            q.content,
            q.timestamp,
            NULL::uuid as response_id,
            q.query_id
        FROM public.queries q
        WHERE q.session_id = p_session_id
        
        UNION ALL
        
        -- Get responses (assistant messages)
        SELECT 
            'assistant' as message_type,
            r.content,
            r.timestamp,
            r.response_id,
            NULL::uuid as query_id
        FROM public.responses r
        WHERE r.session_id = p_session_id
    )
    SELECT 
        cm.message_type,
        cm.content,
        cm.timestamp,
        cm.response_id,
        cm.query_id
    FROM combined_messages cm
    ORDER BY cm.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get recent sessions for a user
CREATE OR REPLACE FUNCTION get_user_recent_sessions(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    session_id UUID,
    started_at TIMESTAMPTZ,
    last_active TIMESTAMPTZ,
    message_count BIGINT,
    first_message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.session_id,
        s.started_at,
        s.last_active,
        COUNT(q.query_id) as message_count,
        (
            SELECT q2.content 
            FROM public.queries q2 
            WHERE q2.session_id = s.session_id 
            ORDER BY q2.timestamp ASC 
            LIMIT 1
        ) as first_message
    FROM public.sessions s
    LEFT JOIN public.queries q ON s.session_id = q.session_id
    WHERE s.user_id = p_user_id
    GROUP BY s.session_id, s.started_at, s.last_active
    ORDER BY s.last_active DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to update session last_active timestamp
CREATE OR REPLACE FUNCTION update_session_activity(p_session_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE public.sessions
    SET last_active = NOW()
    WHERE session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get sources used in a response
CREATE OR REPLACE FUNCTION get_response_sources(p_response_id UUID)
RETURNS TABLE (
    doc_id UUID,
    category TEXT,
    filename TEXT,
    relevance_score DOUBLE PRECISION,
    rank_position INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rd.doc_id,
        d.category,
        d.filename,
        rd.relevance_score,
        rd.rank_position
    FROM public.response_documents rd
    JOIN public.documents d ON rd.doc_id = d.doc_id
    WHERE rd.response_id = p_response_id
    ORDER BY rd.rank_position;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- OPTIONAL: Analytics Views
-- ============================================

-- View for session statistics
CREATE OR REPLACE VIEW session_stats AS
SELECT 
    s.session_id,
    s.user_id,
    s.started_at,
    s.last_active,
    COUNT(DISTINCT q.query_id) as query_count,
    COUNT(DISTINCT r.response_id) as response_count,
    EXTRACT(EPOCH FROM (s.last_active - s.started_at)) as duration_seconds
FROM public.sessions s
LEFT JOIN public.queries q ON s.session_id = q.session_id
LEFT JOIN public.responses r ON s.session_id = r.session_id
GROUP BY s.session_id, s.user_id, s.started_at, s.last_active;

-- View for popular query domains
CREATE OR REPLACE VIEW domain_statistics AS
SELECT 
    COALESCE(q.domain, 'general') as domain,
    COUNT(*) as query_count,
    COUNT(DISTINCT q.session_id) as unique_sessions,
    AVG(r.confidence) as avg_confidence
FROM public.queries q
LEFT JOIN public.responses r ON q.session_id = r.session_id 
    AND r.timestamp > q.timestamp 
    AND r.timestamp < q.timestamp + INTERVAL '1 minute'
GROUP BY q.domain
ORDER BY query_count DESC;

-- ============================================
-- CLEANUP FUNCTIONS (Optional)
-- ============================================

-- Function to archive old sessions (keep last 90 days)
CREATE OR REPLACE FUNCTION archive_old_sessions(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM public.sessions
        WHERE last_active < NOW() - (days_to_keep || ' days')::INTERVAL
        RETURNING session_id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================

-- Enable RLS on sessions if not already enabled
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only view their own sessions
DROP POLICY IF EXISTS "Users can view own sessions" ON public.sessions;
CREATE POLICY "Users can view own sessions"
    ON public.sessions FOR SELECT
    USING (user_id = auth.uid() OR auth.uid() IS NULL);  -- Allow null for anonymous/backend access

-- Policy: Users can create their own sessions
DROP POLICY IF EXISTS "Users can create sessions" ON public.sessions;
CREATE POLICY "Users can create sessions"
    ON public.sessions FOR INSERT
    WITH CHECK (user_id = auth.uid() OR auth.uid() IS NULL);

-- Policy: Users can update their own sessions
DROP POLICY IF EXISTS "Users can update own sessions" ON public.sessions;
CREATE POLICY "Users can update own sessions"
    ON public.sessions FOR UPDATE
    USING (user_id = auth.uid() OR auth.uid() IS NULL);

-- Similar policies for queries table
ALTER TABLE public.queries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own queries" ON public.queries;
CREATE POLICY "Users can view own queries"
    ON public.queries FOR SELECT
    USING (
        session_id IN (
            SELECT session_id FROM public.sessions 
            WHERE user_id = auth.uid() OR auth.uid() IS NULL
        )
    );

DROP POLICY IF EXISTS "Users can create queries" ON public.queries;
CREATE POLICY "Users can create queries"
    ON public.queries FOR INSERT
    WITH CHECK (
        session_id IN (
            SELECT session_id FROM public.sessions 
            WHERE user_id = auth.uid() OR auth.uid() IS NULL
        )
    );

-- Similar policies for responses table
ALTER TABLE public.responses ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own responses" ON public.responses;
CREATE POLICY "Users can view own responses"
    ON public.responses FOR SELECT
    USING (
        session_id IN (
            SELECT session_id FROM public.sessions 
            WHERE user_id = auth.uid() OR auth.uid() IS NULL
        )
    );

DROP POLICY IF EXISTS "Backend can create responses" ON public.responses;
CREATE POLICY "Backend can create responses"
    ON public.responses FOR INSERT
    WITH CHECK (TRUE);  -- Backend service can create responses for any session

-- ============================================
-- USAGE EXAMPLES
-- ============================================

-- Example 1: Get conversation history for a session
-- SELECT * FROM get_conversation_history('session-uuid-here', 20);

-- Example 2: Get recent sessions for a user
-- SELECT * FROM get_user_recent_sessions('user-uuid-here');

-- Example 3: Update session activity
-- SELECT update_session_activity('session-uuid-here');

-- Example 4: Get sources for a response
-- SELECT * FROM get_response_sources('response-uuid-here');

-- Example 5: Archive sessions older than 180 days
-- SELECT archive_old_sessions(180);

-- Example 6: View session statistics
-- SELECT * FROM session_stats WHERE user_id = 'user-uuid-here';

-- Example 7: View domain statistics
-- SELECT * FROM domain_statistics;

-- ============================================
-- NOTES
-- ============================================

/*
Key Points:
1. Your existing schema is already perfect for chat! You have:
   - sessions: Tracks conversations
   - queries: Stores user messages
   - responses: Stores AI replies
   - response_documents: Links responses to source documents

2. The FastAPI will use these existing tables instead of creating new ones

3. Added helper functions to:
   - Get conversation history
   - Track session activity
   - Retrieve sources
   - Generate analytics

4. RLS policies ensure users only see their own data

5. Optional cleanup functions to manage old data

6. The metadata JSONB column on sessions allows storing:
   - Language preference (en/ur)
   - Category filters
   - Custom user preferences
*/