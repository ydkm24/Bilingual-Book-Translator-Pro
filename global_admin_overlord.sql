-- Global Admin Overlord (Universal RLS Bypass)
-- Grants users with 'is_admin = true' absolute power over all tables.

-- FORUMS: Topics
DROP POLICY IF EXISTS "Admin Topics Delete" ON forum_topics;
CREATE POLICY "Admin Topics Delete" ON forum_topics FOR DELETE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

DROP POLICY IF EXISTS "Admin Topics Update" ON forum_topics;
CREATE POLICY "Admin Topics Update" ON forum_topics FOR UPDATE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- FORUMS: Posts
DROP POLICY IF EXISTS "Admin Posts Delete" ON forum_posts;
CREATE POLICY "Admin Posts Update" ON forum_posts FOR UPDATE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

DROP POLICY IF EXISTS "Admin Posts Delete" ON forum_posts;
CREATE POLICY "Admin Posts Delete" ON forum_posts FOR DELETE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- FEED & SOCIAL
DROP POLICY IF EXISTS "Admin Feed Delete" ON activity_feed;
CREATE POLICY "Admin Feed Delete" ON activity_feed FOR DELETE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- BOOKS / PROJECTS (Arbiter control over metadata)
DROP POLICY IF EXISTS "Admin Books Update" ON books;
CREATE POLICY "Admin Books Update" ON books FOR UPDATE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

DROP POLICY IF EXISTS "Admin Books Delete" ON books;
CREATE POLICY "Admin Books Delete" ON books FOR DELETE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- PROFILES (Admin can moderate bios/usernames)
DROP POLICY IF EXISTS "Admin Profiles Update" ON profiles;
CREATE POLICY "Admin Profiles Update" ON profiles FOR UPDATE USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);
