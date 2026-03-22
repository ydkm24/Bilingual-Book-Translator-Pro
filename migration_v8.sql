-- 1. Add is_public to forum_topics (default to true)
ALTER TABLE forum_topics ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT TRUE;

-- 2. Update existing topics to be public
UPDATE forum_topics SET is_public = TRUE WHERE is_public IS NULL;

-- 3. Update Policy for Topics (Allow selective visibility)
DROP POLICY IF EXISTS "Public Topics Read" ON forum_topics;
CREATE POLICY "Public Topics Read" ON forum_topics FOR SELECT USING (
    is_public = TRUE OR 
    auth.uid() = author_id OR 
    EXISTS (SELECT 1 FROM friends WHERE user_id = forum_topics.author_id AND friend_id = auth.uid())
);
