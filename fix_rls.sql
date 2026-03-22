-- Categories
DROP POLICY IF EXISTS "Public Categories Read" ON forum_categories;
CREATE POLICY "Public Categories Read" ON forum_categories FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Categories Insert" ON forum_categories;
CREATE POLICY "Auth Categories Insert" ON forum_categories FOR INSERT WITH CHECK (auth.role() = 'authenticated');

DROP POLICY IF EXISTS "Auth Categories Delete" ON forum_categories;
CREATE POLICY "Auth Categories Delete" ON forum_categories FOR DELETE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- Topics
DROP POLICY IF EXISTS "Public Topics Read" ON forum_topics;
CREATE POLICY "Public Topics Read" ON forum_topics FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Topics Insert" ON forum_topics;
CREATE POLICY "Auth Topics Insert" ON forum_topics FOR INSERT WITH CHECK (auth.uid() = author_id);

DROP POLICY IF EXISTS "Mod Topics Update" ON forum_topics;
CREATE POLICY "Mod Topics Update" ON forum_topics FOR UPDATE USING (
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM forum_moderators WHERE user_id = auth.uid() AND topic_id = id)
);

-- Posts
DROP POLICY IF EXISTS "Public Posts Read" ON forum_posts;
CREATE POLICY "Public Posts Read" ON forum_posts FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Posts Insert" ON forum_posts;
CREATE POLICY "Auth Posts Insert" ON forum_posts FOR INSERT WITH CHECK (
    auth.uid() = author_id AND 
    NOT EXISTS (SELECT 1 FROM blocked_forum_users WHERE user_id = auth.uid() AND topic_id = forum_posts.topic_id)
);

-- Moderators
DROP POLICY IF EXISTS "Public Moderators Read" ON forum_moderators;
CREATE POLICY "Public Moderators Read" ON forum_moderators FOR SELECT USING (true);

DROP POLICY IF EXISTS "Auth Moderators Insert" ON forum_moderators;
CREATE POLICY "Auth Moderators Insert" ON forum_moderators FOR INSERT WITH CHECK (
    auth.uid() = user_id OR 
    EXISTS (SELECT 1 FROM forum_moderators WHERE user_id = auth.uid() AND topic_id = forum_moderators.topic_id AND role = 'admin')
);
