-- Fix Forum Relationships for PostgREST
-- This allows automated profile joins by pointing FKs to public.profiles instead of auth.users

-- 1. Forum Categories
ALTER TABLE forum_categories DROP CONSTRAINT IF EXISTS forum_categories_author_id_fkey;
ALTER TABLE forum_categories 
ADD CONSTRAINT forum_categories_author_id_fkey 
FOREIGN KEY (author_id) REFERENCES profiles(id) ON DELETE SET NULL;

-- 2. Forum Topics
ALTER TABLE forum_topics DROP CONSTRAINT IF EXISTS forum_topics_author_id_fkey;
ALTER TABLE forum_topics 
ADD CONSTRAINT forum_topics_author_id_fkey 
FOREIGN KEY (author_id) REFERENCES profiles(id) ON DELETE SET NULL;

-- 3. Forum Posts
ALTER TABLE forum_posts DROP CONSTRAINT IF EXISTS forum_posts_author_id_fkey;
ALTER TABLE forum_posts 
ADD CONSTRAINT forum_posts_author_id_fkey 
FOREIGN KEY (author_id) REFERENCES profiles(id) ON DELETE SET NULL;

-- 4. Forum Moderators
ALTER TABLE forum_moderators DROP CONSTRAINT IF EXISTS forum_moderators_user_id_fkey;
ALTER TABLE forum_moderators 
ADD CONSTRAINT forum_moderators_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES profiles(id) ON DELETE CASCADE;

-- 5. Blocked Users
ALTER TABLE blocked_forum_users DROP CONSTRAINT IF EXISTS blocked_forum_users_user_id_fkey;
ALTER TABLE blocked_forum_users 
ADD CONSTRAINT blocked_forum_users_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES profiles(id) ON DELETE CASCADE;

-- FINAL STEP: RELOAD SCHEMA CACHE
-- (Supabase does this automatically when DDL changes, but good to know)
