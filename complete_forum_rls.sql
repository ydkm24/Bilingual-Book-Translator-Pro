-- Complete Forum RLS (Phase 7.5: Deletion Support)
-- Adds missing DELETE policies for topics, posts, and moderators.

-- 1. Topics Delete
DROP POLICY IF EXISTS "Auth Topics Delete" ON forum_topics;
CREATE POLICY "Auth Topics Delete" ON forum_topics FOR DELETE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- 2. Posts Delete
DROP POLICY IF EXISTS "Auth Posts Delete" ON forum_posts;
CREATE POLICY "Auth Posts Delete" ON forum_posts FOR DELETE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM forum_topics WHERE id = topic_id AND author_id = auth.uid()) OR
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- 3. Moderators Delete
DROP POLICY IF EXISTS "Auth Moderators Delete" ON forum_moderators;
CREATE POLICY "Auth Moderators Delete" ON forum_moderators FOR DELETE USING (
    auth.uid() = user_id OR 
    EXISTS (SELECT 1 FROM forum_moderators WHERE user_id = auth.uid() AND topic_id = forum_moderators.topic_id AND role = 'admin')
);

-- 4. activity_feed Delete (Allow users to delete their own activity notifications)
DROP POLICY IF EXISTS "Auth Feed Delete" ON activity_feed;
CREATE POLICY "Auth Feed Delete" ON activity_feed FOR DELETE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- 5. Posts Update (NEW: Author only)
DROP POLICY IF EXISTS "Auth Posts Update" ON forum_posts;
CREATE POLICY "Auth Posts Update" ON forum_posts FOR UPDATE USING (auth.uid() = author_id);
