-- Neural Arbiter Evolution: Migration v4.3.1
-- Optimized for: Featured Curation, Processing Control, and Centralized Thaw

-- 1. BOOK CURATION
ALTER TABLE books ADD COLUMN IF NOT EXISTS is_featured BOOLEAN DEFAULT FALSE;
COMMENT ON COLUMN books.is_featured IS 'True if the Neural Arbiter has promoted this work to the Global Spotlight.';

-- 2. PROCESSING CONTROL
ALTER TABLE pages ADD COLUMN IF NOT EXISTS is_skipped BOOLEAN DEFAULT FALSE;
COMMENT ON COLUMN pages.is_skipped IS 'True if the user/arbiter decided to skip OCR/Translation for this specific page.';

-- 3. USER MANAGEMENT (Ensuring column exists for Audit Log)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_frozen BOOLEAN DEFAULT FALSE;

-- 4. RLS UPDATES (Allow Arbiter to update featured status)
-- Note: Assuming Arbiter already has bypass via is_admin, but good to be explicit.
DROP POLICY IF EXISTS "Arbiter Manage Featured" ON books;
CREATE POLICY "Arbiter Manage Featured" ON books 
FOR UPDATE USING (
    (SELECT is_admin FROM profiles WHERE id = auth.uid()) = TRUE
);

-- 5. AUDIT LOG INITIALIZATION (If not already present from arbiter_engine_setup)
CREATE TABLE IF NOT EXISTS arbiter_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    arbiter_id UUID REFERENCES auth.users(id),
    action_type TEXT NOT NULL, -- 'FREEZE_USER', 'DELETE_POST', 'FEATURE_BOOK', 'KILL_SWITCH', etc.
    target_id TEXT, -- ID of the affected user/post/book
    target_name TEXT, -- Human readable name (e.g. username, book title)
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE arbiter_audit_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Arbiter View Logs" ON arbiter_audit_log;
CREATE POLICY "Arbiter View Logs" ON arbiter_audit_log FOR SELECT USING (
    (SELECT is_admin FROM profiles WHERE id = auth.uid()) = TRUE
);
