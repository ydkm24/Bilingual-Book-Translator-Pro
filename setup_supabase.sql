-- Bilingual Book Translator Pro: Master Setup Script (v1.4.2)
-- Run this in your Supabase SQL Editor. 
-- This script is "Safe" (Re-runnable): It will not duplicate data or crash if run multiple times.

-- ==========================================
-- 1. EXTENSIONS
-- ==========================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================
-- 2. TABLES
-- ==========================================

-- Books Table
CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    total_pages INTEGER NOT NULL,
    language TEXT NOT NULL,
    category TEXT DEFAULT 'Uncategorized',
    is_public BOOLEAN DEFAULT FALSE,
    is_read_only BOOLEAN DEFAULT TRUE,
    owner_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    owner_username TEXT,
    likes INTEGER DEFAULT 0,
    active_editor_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    active_editor_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pages Table
CREATE TABLE IF NOT EXISTS pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    original TEXT,
    english TEXT,
    literal TEXT,
    is_image BOOLEAN DEFAULT FALSE,
    is_cover BOOLEAN DEFAULT FALSE,
    is_centered BOOLEAN DEFAULT FALSE,
    is_rtl_page BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(book_id, page_index)
);

-- Profiles Table
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    avatar_url TEXT,
    verified_badge BOOLEAN DEFAULT FALSE,
    bio TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Permission Requests Table
CREATE TABLE IF NOT EXISTS permission_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    requester_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    requester_username TEXT NOT NULL,
    status TEXT DEFAULT 'pending', 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(book_id, requester_id)
);

-- Comments Table
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    username TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Book Likes Table (New in v1.4.2)
CREATE TABLE IF NOT EXISTS book_likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(book_id, user_id)
);

-- ==========================================
-- 3. ENABLE RLS
-- ==========================================
ALTER TABLE books ENABLE ROW LEVEL SECURITY;
ALTER TABLE pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE permission_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE book_likes ENABLE ROW LEVEL SECURITY;

-- ==========================================
-- 4. POLICIES (DROP & RECREATE FOR SAFETY)
-- ==========================================

-- BOOKS
DROP POLICY IF EXISTS "Select Books" ON books;
CREATE POLICY "Select Books" ON books FOR SELECT USING (is_public = true OR auth.uid() = owner_id);

DROP POLICY IF EXISTS "Insert Books" ON books;
CREATE POLICY "Insert Books" ON books FOR INSERT WITH CHECK (auth.uid() = owner_id);

DROP POLICY IF EXISTS "Update/Delete Books" ON books;
CREATE POLICY "Update/Delete Books" ON books FOR ALL USING (auth.uid() = owner_id);

-- PAGES
DROP POLICY IF EXISTS "Select Pages" ON pages;
CREATE POLICY "Select Pages" ON pages FOR SELECT USING (EXISTS (SELECT 1 FROM books WHERE id = book_id AND (is_public = true OR auth.uid() = owner_id)));

DROP POLICY IF EXISTS "Modify Pages" ON pages;
CREATE POLICY "Modify Pages" ON pages FOR ALL USING (EXISTS (SELECT 1 FROM books WHERE id = book_id AND auth.uid() = owner_id));

-- PROFILES
DROP POLICY IF EXISTS "View Profiles" ON profiles;
CREATE POLICY "View Profiles" ON profiles FOR SELECT USING (true);

DROP POLICY IF EXISTS "Modify Own Profile" ON profiles;
CREATE POLICY "Modify Own Profile" ON profiles FOR ALL USING (auth.uid() = id);

-- REQUESTS
DROP POLICY IF EXISTS "View Requests" ON permission_requests;
CREATE POLICY "View Requests" ON permission_requests FOR SELECT USING (auth.uid() = requester_id OR EXISTS (SELECT 1 FROM books WHERE id = book_id AND owner_id = auth.uid()));

DROP POLICY IF EXISTS "Create Request" ON permission_requests;
CREATE POLICY "Create Request" ON permission_requests FOR INSERT WITH CHECK (auth.uid() = requester_id);

-- COMMENTS
DROP POLICY IF EXISTS "Select Comments" ON comments;
CREATE POLICY "Select Comments" ON comments FOR SELECT USING (EXISTS (SELECT 1 FROM books WHERE id = book_id AND (is_public = true OR auth.uid() = owner_id)));

DROP POLICY IF EXISTS "Insert Comments" ON comments;
CREATE POLICY "Insert Comments" ON comments FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Delete Comments" ON comments;
CREATE POLICY "Delete Comments" ON comments FOR DELETE USING (auth.uid() = user_id);

-- LIKES
DROP POLICY IF EXISTS "Select Book Likes" ON book_likes;
CREATE POLICY "Select Book Likes" ON book_likes FOR SELECT USING (true);

DROP POLICY IF EXISTS "Insert Book Likes" ON book_likes;
CREATE POLICY "Insert Book Likes" ON book_likes FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Delete Book Likes" ON book_likes;
CREATE POLICY "Delete Book Likes" ON book_likes FOR DELETE USING (auth.uid() = user_id);
