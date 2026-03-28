-- SQL Migration: Library Bookmarks & Progression
-- Run this in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS public.user_bookmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
    last_page_index INT DEFAULT 0,
    is_bookmarked BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, book_id)
);

-- Enable RLS
ALTER TABLE public.user_bookmarks ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can manage their own bookmarks" 
ON public.user_bookmarks FOR ALL 
USING (auth.uid() = user_id);

-- Search Index for performance
CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON public.user_bookmarks(user_id);
CREATE INDEX IF NOT EXISTS idx_bookmarks_combined ON public.user_bookmarks(user_id, book_id);
