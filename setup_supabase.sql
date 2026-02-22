-- Run this in your Supabase SQL Editor to set up the database schema

-- 1. Create the 'books' table
CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    total_pages INTEGER NOT NULL,
    language TEXT NOT NULL,
    category TEXT DEFAULT 'Uncategorized',  -- SPRINT 33: Folder system
    is_public BOOLEAN DEFAULT FALSE,
    is_read_only BOOLEAN DEFAULT TRUE,      -- SPRINT 33: Permissions
    owner_id UUID REFERENCES auth.users(id) ON DELETE SET NULL, -- SPRINT 34: Ownership
    owner_username TEXT,                                        -- SPRINT 34: Ownership display
    likes INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Note: Run these ALTER commands manually if updating an existing database
-- ALTER TABLE books ADD COLUMN category TEXT DEFAULT 'Uncategorized';
-- ALTER TABLE books ADD COLUMN is_read_only BOOLEAN DEFAULT TRUE;

-- 2. Create the 'pages' table
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

-- 3. Enable RLS (Row Level Security)
ALTER TABLE books ENABLE ROW LEVEL SECURITY;
ALTER TABLE pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE permission_requests ENABLE ROW LEVEL SECURITY;

-- ### BOOKS POLICIES ###

-- Anyone can see public books OR their own private books
CREATE POLICY "Select Books" ON books FOR SELECT 
USING (is_public = true OR auth.uid() = owner_id);

-- Only authenticated users can create books
CREATE POLICY "Insert Books" ON books FOR INSERT 
WITH CHECK (auth.uid() = owner_id);

-- Only owners can update or delete their books
CREATE POLICY "Update/Delete Books" ON books FOR ALL
USING (auth.uid() = owner_id)
WITH CHECK (auth.uid() = owner_id);

-- ### PAGES POLICIES ###

-- Access pages if the book itself is accessible
CREATE POLICY "Select Pages" ON pages FOR SELECT 
USING (EXISTS (SELECT 1 FROM books WHERE id = book_id AND (is_public = true OR auth.uid() = owner_id)));

-- Only owners can insert/update pages (via book ownership)
CREATE POLICY "Modify Pages" ON pages FOR ALL
USING (EXISTS (SELECT 1 FROM books WHERE id = book_id AND auth.uid() = owner_id))
WITH CHECK (EXISTS (SELECT 1 FROM books WHERE id = book_id AND auth.uid() = owner_id));

-- ### PROFILES POLICIES ###

-- Profiles are publically readable (for avatars/usernames)
CREATE POLICY "View Profiles" ON profiles FOR SELECT USING (true);

-- Users can only modify their own profile
CREATE POLICY "Modify Own Profile" ON profiles FOR ALL 
USING (auth.uid() = id) 
WITH CHECK (auth.uid() = id);

-- ### PERMISSION REQUESTS POLICIES ###

-- Requesters can see their own requests, owners can see requests for their books
CREATE POLICY "View Requests" ON permission_requests FOR SELECT
USING (auth.uid() = requester_id OR EXISTS (SELECT 1 FROM books WHERE id = book_id AND owner_id = auth.uid()));

-- Only authenticated users can create a request
CREATE POLICY "Create Request" ON permission_requests FOR INSERT
WITH CHECK (auth.uid() = requester_id);

-- Only book owners can Approve/Deny (Update)
CREATE POLICY "Moderate Requests" ON permission_requests FOR UPDATE
USING (EXISTS (SELECT 1 FROM books WHERE id = book_id AND owner_id = auth.uid()));

-- 4. Create the 'profiles' table for Usernames (SPRINT 34)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username TEXT UNIQUE NOT NULL,
    email TEXT, -- SPRINT 36: Used to resolve "Sign In with Username"
    is_admin BOOLEAN DEFAULT FALSE,
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Note: Run this ALTER command manually if updating an existing database for Sprint 36
-- ALTER TABLE profiles ADD COLUMN email TEXT;

-- 5. Create the 'permission_requests' table (SPRINT 34)
CREATE TABLE IF NOT EXISTS permission_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    requester_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    requester_username TEXT NOT NULL,
    status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'denied'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(book_id, requester_id)
);

-- 6. Storage Bucket Setup
-- Note: You should manually create a bucket named 'avatars' in the Supabase Dashboard
-- with Public Access enabled. Then run the following commands to grant upload permission:

-- Allow authenticated users to upload new avatars
CREATE POLICY "Allow authenticated uploads"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK ( bucket_id = 'avatars' );

-- Allow users to update their own existing avatars
CREATE POLICY "Allow authenticated updates"
ON storage.objects FOR UPDATE
TO authenticated
USING ( bucket_id = 'avatars' AND owner_id = (select auth.uid())::text );

-- Allow public viewing of avatars
CREATE POLICY "Allow public viewing"
ON storage.objects FOR SELECT
TO public
USING ( bucket_id = 'avatars' );
