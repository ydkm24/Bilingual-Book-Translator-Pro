-- ==========================================================
-- MASTER FORUM SETUP (Strictly Forums)
-- ==========================================================
-- This script consolidates all forum tables, relationships, 
-- and forum-specific RLS policies. 
-- Run this in the Supabase SQL Editor.

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

DROP POLICY IF EXISTS "Admin Categories Manage" ON forum_categories;
CREATE POLICY "Admin Categories Manage" ON forum_categories FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- 4. POLICIES: FORUM TOPICS -------------------------------

DROP POLICY IF EXISTS "Public Topics Read" ON forum_topics;
CREATE POLICY "Public Topics Read" ON forum_topics FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Topics Insert" ON forum_topics;
CREATE POLICY "Auth Topics Insert" ON forum_topics FOR INSERT WITH CHECK (auth.uid() = author_id);

DROP POLICY IF EXISTS "Auth Topics Delete" ON forum_topics;
CREATE POLICY "Auth Topics Delete" ON forum_topics FOR DELETE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

DROP POLICY IF EXISTS "Mod Topics Update" ON forum_topics;
CREATE POLICY "Mod Topics Update" ON forum_topics FOR UPDATE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM forum_moderators WHERE user_id = auth.uid() AND topic_id = id)
);

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
    EXISTS (SELECT 1 FROM forum_topics WHERE id = topic_id AND author_id = auth.uid()) OR
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- 6. POLICIES: MODERATORS & BLOCKED ----------------------

DROP POLICY IF EXISTS "Public Moderators Read" ON forum_moderators;
CREATE POLICY "Public Moderators Read" ON forum_moderators FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Moderators Insert" ON forum_moderators;
CREATE POLICY "Auth Moderators Insert" ON forum_moderators FOR INSERT WITH CHECK (
    auth.uid() = user_id OR 
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true) 
);

DROP POLICY IF EXISTS "Admin Moderators Manage" ON forum_moderators;
CREATE POLICY "Admin Moderators Manage" ON forum_moderators FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);


-- ==========================================================
-- SETUP COMPLETE.
-- ==========================================================
