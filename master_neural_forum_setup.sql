-- Run this in the Supabase SQL Editor.

-- 0. SYSTEM FUNCTIONS (To avoid recursion) ----------------

CREATE OR REPLACE FUNCTION is_arbiter() 
RETURNS BOOLEAN AS $$
  SELECT is_admin FROM profiles WHERE id = auth.uid();
$$ LANGUAGE sql SECURITY DEFINER SET search_path = public;

-- 1. TABLES -----------------------------------------------

CREATE TABLE IF NOT EXISTS forum_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    icon TEXT,
    is_system BOOLEAN DEFAULT FALSE,
    author_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forum_topics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category_id UUID REFERENCES forum_categories(id) ON DELETE CASCADE,
    author_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    tags JSONB DEFAULT '[]',
    is_pinned BOOLEAN DEFAULT FALSE,
    is_locked BOOLEAN DEFAULT FALSE,
    is_system BOOLEAN DEFAULT FALSE,
    view_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forum_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic_id UUID REFERENCES forum_topics(id) ON DELETE CASCADE,
    author_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forum_moderators (
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES forum_topics(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('admin', 'moderator')),
    PRIMARY KEY (user_id, topic_id)
);

CREATE TABLE IF NOT EXISTS blocked_forum_users (
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES forum_topics(id) ON DELETE CASCADE,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, topic_id)
);

-- 2. SECURITY (RLS ENABLE) --------------------------------

ALTER TABLE forum_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE forum_topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE forum_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE forum_moderators ENABLE ROW LEVEL SECURITY;
ALTER TABLE blocked_forum_users ENABLE ROW LEVEL SECURITY;

-- 3. POLICIES: FORUM CATEGORIES ---------------------------

DROP POLICY IF EXISTS "Public Categories Read" ON forum_categories;
CREATE POLICY "Public Categories Read" ON forum_categories FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Categories Insert" ON forum_categories;
CREATE POLICY "Auth Categories Insert" ON forum_categories FOR INSERT WITH CHECK (auth.role() = 'authenticated');

DROP POLICY IF EXISTS "Arbiter Categories All" ON forum_categories;
CREATE POLICY "Arbiter Categories All" ON forum_categories FOR ALL USING (is_arbiter());

-- 4. POLICIES: FORUM TOPICS -------------------------------

DROP POLICY IF EXISTS "Public Topics Read" ON forum_topics;
CREATE POLICY "Public Topics Read" ON forum_topics FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Topics Insert" ON forum_topics;
CREATE POLICY "Auth Topics Insert" ON forum_topics FOR INSERT WITH CHECK (auth.uid() = author_id);

DROP POLICY IF EXISTS "Arbiter Topics All" ON forum_topics;
CREATE POLICY "Arbiter Topics All" ON forum_topics FOR ALL USING (is_arbiter());

DROP POLICY IF EXISTS "Auth Topics Delete" ON forum_topics;
CREATE POLICY "Auth Topics Delete" ON forum_topics FOR DELETE USING (auth.uid() = author_id);

-- 5. POLICIES: FORUM POSTS --------------------------------

DROP POLICY IF EXISTS "Public Posts Read" ON forum_posts;
CREATE POLICY "Public Posts Read" ON forum_posts FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Posts Insert" ON forum_posts;
CREATE POLICY "Auth Posts Insert" ON forum_posts FOR INSERT WITH CHECK (
    auth.uid() = author_id AND 
    NOT EXISTS (SELECT 1 FROM blocked_forum_users WHERE user_id = auth.uid() AND topic_id = forum_posts.topic_id)
);

DROP POLICY IF EXISTS "Auth Posts Update" ON forum_posts;
CREATE POLICY "Auth Posts Update" ON forum_posts FOR UPDATE USING (auth.uid() = author_id);

DROP POLICY IF EXISTS "Auth Posts Delete" ON forum_posts;
CREATE POLICY "Auth Posts Delete" ON forum_posts FOR DELETE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM forum_topics WHERE id = topic_id AND author_id = auth.uid())
);

DROP POLICY IF EXISTS "Arbiter Posts All" ON forum_posts;
CREATE POLICY "Arbiter Posts All" ON forum_posts FOR ALL USING (is_arbiter());

-- 6. POLICIES: MODERATORS & BLOCKED ----------------------

DROP POLICY IF EXISTS "Public Moderators Read" ON forum_moderators;
CREATE POLICY "Public Moderators Read" ON forum_moderators FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Moderators Insert" ON forum_moderators;
CREATE POLICY "Auth Moderators Insert" ON forum_moderators FOR INSERT WITH CHECK (
    auth.uid() = user_id OR is_arbiter()
);

DROP POLICY IF EXISTS "Arbiter Moderators All" ON forum_moderators;
CREATE POLICY "Arbiter Moderators All" ON forum_moderators FOR ALL USING (is_arbiter());

-- 7. INITIAL DATA (Run once) -----------------------------

-- ONLY run these if you want to seed default categories
-- INSERT INTO forum_categories (id, name, description, icon, is_system) VALUES
-- (gen_random_uuid(), 'Official Archives', 'Patch Notes and Announcements.', '📜', TRUE),
-- (gen_random_uuid(), 'Community Lounge', 'Talk about anything!', '💬', FALSE)
-- ON CONFLICT (id) DO NOTHING;

-- ==========================================================
-- SETUP COMPLETE.
-- ==========================================================
