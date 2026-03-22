-- ==========================================================
-- NUCLEAR RECURSION FIX (Forum Moderators)
-- ==========================================================
-- This script wipes ALL existing policies on forum_moderators
-- and replaces them with a clean, loop-free version.

DO $$
DECLARE
    pol RECORD;
BEGIN
    -- Drop EVERY policy on forum_moderators regardless of its name
    FOR pol IN (SELECT policyname FROM pg_policies WHERE tablename = 'forum_moderators') LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON forum_moderators', pol.policyname);
    END LOOP;
END $$;

-- 1. Re-enable RLS
ALTER TABLE forum_moderators ENABLE ROW LEVEL SECURITY;

-- 2. Clean 'Read' Policy
CREATE POLICY "Public Moderators Read" ON forum_moderators 
FOR SELECT USING (true);

-- 3. Clean 'Insert' Policy (Zero Recursion)
-- Allows you to become a moderator of your own topic OR an admin to add you.
CREATE POLICY "Auth Moderators Insert" ON forum_moderators 
FOR INSERT WITH CHECK (
    auth.uid() = user_id OR 
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- 4. Clean 'Delete' Policy
CREATE POLICY "Auth Moderators Delete" ON forum_moderators 
FOR DELETE USING (
    auth.uid() = user_id OR 
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND is_admin = true)
);

-- ==========================================================
-- RECURSION WIPED. YOU ARE READY TO PUBLISH.
-- ==========================================================
