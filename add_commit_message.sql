-- Migration: Add last_commit_message column to books table
-- Run this in Supabase SQL Editor

ALTER TABLE books ADD COLUMN IF NOT EXISTS last_commit_message TEXT;
