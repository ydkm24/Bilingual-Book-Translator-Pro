-- Run this in the Supabase SQL Editor.

-- 0. SYSTEM FUNCTIONS -------------------------------------

CREATE OR REPLACE FUNCTION is_arbiter() 
RETURNS BOOLEAN AS $$
  SELECT is_admin FROM profiles WHERE id = auth.uid();
$$ LANGUAGE sql SECURITY DEFINER SET search_path = public;

-- 1. ACTIVITY FEED ----------------------------------------
-- Allows purging any entry from the global neural stream.

DROP POLICY IF EXISTS "Arbiter Feed All" ON activity_feed;
CREATE POLICY "Arbiter Feed All" ON activity_feed FOR ALL USING (is_arbiter());

-- 2. GLOBAL LIBRARY (BOOKS) -------------------------------
-- Allows moderating project metadata and expunging publications.

DROP POLICY IF EXISTS "Arbiter Books All" ON books;
CREATE POLICY "Arbiter Books All" ON books FOR ALL USING (is_arbiter());

-- 3. USER PROFILES ----------------------------------------
-- Allows resetting usernames, bios, or 'Freezing' accounts.

DROP POLICY IF EXISTS "Arbiter Profiles Update" ON profiles;
CREATE POLICY "Arbiter Profiles Update" ON profiles FOR UPDATE USING (is_arbiter());

-- 4. DIRECT MESSAGES (RESTRICTED) -------------------------
-- Uncomment the below only if you want absolute DM oversight.

-- DROP POLICY IF EXISTS "Arbiter DMs All" ON direct_messages;
-- CREATE POLICY "Arbiter DMs All" ON direct_messages FOR ALL USING (is_arbiter());

-- ==========================================================
-- GLOBAL AUTHORITY ESTABLISHED.
-- ==========================================================
