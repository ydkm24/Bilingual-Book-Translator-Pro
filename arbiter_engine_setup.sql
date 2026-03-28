-- ==========================================================
-- NEURAL ARBITER ENGINE (Backend Infrastructure)
-- ==========================================================
-- This script establishes the hardware for God Mode:
-- 1. Global Broadcasts (Popups/Banners)
-- 2. Neural Audit Log (History of Arbiter actions)
-- 3. App Configuration (Kill Switch, Versioning)
-- ==========================================================

-- 1. APP CONFIGURATION TABLE ------------------------------
CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by UUID REFERENCES profiles(id)
);

-- Initialize defaults
INSERT INTO app_config (key, value) VALUES ('kill_switch', 'false') ON CONFLICT DO NOTHING;
INSERT INTO app_config (key, value) VALUES ('latest_version', '4.2.0') ON CONFLICT DO NOTHING;
INSERT INTO app_config (key, value) VALUES ('maintenance_message', 'The Neural Link is currently undergoing optimization. Please standby.') ON CONFLICT DO NOTHING;

-- 2. GLOBAL BROADCASTS ------------------------------------
CREATE TABLE IF NOT EXISTS global_broadcasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES profiles(id),
    type TEXT NOT NULL, -- 'popup', 'banner', 'notif'
    message TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ
);

-- 3. NEURAL AUDIT LOG -------------------------------------
CREATE TABLE IF NOT EXISTS arbiter_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    arbiter_id UUID REFERENCES profiles(id),
    action TEXT NOT NULL,
    target_id TEXT, -- User ID, Forum ID, etc.
    details JSONB,
    undone BOOLEAN DEFAULT false
);

-- 4. SECURITY (Arbiter Only) ------------------------------
ALTER TABLE app_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE global_broadcasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE arbiter_audit_log ENABLE ROW LEVEL SECURITY;

-- Utility check (re-using the logic from master_forum_setup)
CREATE OR REPLACE FUNCTION is_arbiter_v2() 
RETURNS BOOLEAN AS $$
  SELECT is_admin FROM profiles WHERE id = auth.uid();
$$ LANGUAGE sql SECURITY DEFINER;

-- Policies
CREATE POLICY "Arbiter Config Read" ON app_config FOR SELECT USING (true);
CREATE POLICY "Arbiter Config Write" ON app_config FOR ALL USING (is_arbiter_v2());

CREATE POLICY "Arbiter Broadcast Read" ON global_broadcasts FOR SELECT USING (true);
CREATE POLICY "Arbiter Broadcast Write" ON global_broadcasts FOR ALL USING (is_arbiter_v2());

CREATE POLICY "Arbiter Audit Read" ON arbiter_audit_log FOR SELECT USING (is_arbiter_v2());
CREATE POLICY "Arbiter Audit Write" ON arbiter_audit_log FOR INSERT WITH CHECK (is_arbiter_v2());
