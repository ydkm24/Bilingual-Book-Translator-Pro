-- Step 0: Elevation to Neural Arbiter
-- Replace 'your_email@example.com' with your actual account email
-- OR replace 'user_uuid' with your actual Profile ID (visible in your Profile URL or DB)

UPDATE profiles 
SET is_admin = true 
WHERE email = 'your_email@example.com';

-- Verify:
SELECT id, username, email, is_admin FROM profiles WHERE is_admin = true;
