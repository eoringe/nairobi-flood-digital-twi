-- ===========================================================================
-- Nairobi Urban Flood Digital Twin - logical schema
-- ===========================================================================
-- Implements the entity-relationship design in Chapter 4. Written to run on
-- BOTH SQLite and PostgreSQL/PostGIS so the project can be demonstrated with
-- zero database installation and later moved to PostGIS without a rewrite.
--
-- Portability rules observed here:
--   * INTEGER PRIMARY KEY autoincrements on SQLite; the PostgreSQL loader
--     rewrites it to GENERATED ALWAYS AS IDENTITY.
--   * Geometry is stored as plain latitude/longitude REAL columns. On PostGIS
--     the loader adds a geometry(Point,4326) column and a GiST index. Storing
--     the ordinates regardless means every query works on both engines.
--   * Timestamps are ISO-8601 TEXT. SQLite has no native timestamp type, and
--     ISO-8601 sorts lexicographically, so comparisons behave identically.
--
-- Gridded arrays (flood probability over 198x252 cells) are NOT stored one row
-- per pixel in normal operation - that would be 49,896 rows per simulation. The
-- array is written to the artefact store and referenced by path. flood_pixels
-- exists for the subset of cells a user explicitly inspects.
-- ===========================================================================

-- ---------------------------------------------------------------- identity --
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    email         TEXT,
    password_hash TEXT NOT NULL,          -- salted hash only (NFR-05)
    role          TEXT NOT NULL DEFAULT 'citizen'
                  CHECK (role IN ('citizen','planner','responder','admin')),
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token      TEXT NOT NULL UNIQUE,
    ip_address TEXT,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(token);

-- ------------------------------------------------------------ meteorology --
CREATE TABLE IF NOT EXISTS rainfall_stations (
    station_id   INTEGER PRIMARY KEY,
    station_name TEXT NOT NULL,
    latitude     REAL NOT NULL,
    longitude    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rainfall_observations (
    obs_id           INTEGER PRIMARY KEY,
    station_id       INTEGER REFERENCES rainfall_stations(station_id),
    observation_time TEXT NOT NULL,
    rainfall_mm      REAL NOT NULL CHECK (rainfall_mm >= 0),
    latitude         REAL,
    longitude        REAL
);
CREATE INDEX IF NOT EXISTS idx_rain_obs_time ON rainfall_observations(observation_time);

CREATE TABLE IF NOT EXISTS temperature_readings (
    reading_id       INTEGER PRIMARY KEY,
    station_id       INTEGER REFERENCES rainfall_stations(station_id),
    observation_time TEXT NOT NULL,
    temperature_c    REAL
);

-- ----------------------------------------------------------------- terrain --
-- One row per prediction-grid cell: 198 x 252 = 49,896 rows, loaded once.
CREATE TABLE IF NOT EXISTS topographic_grid (
    cell_id   INTEGER PRIMARY KEY,
    grid_x    INTEGER NOT NULL,
    grid_y    INTEGER NOT NULL,
    latitude  REAL NOT NULL,
    longitude REAL NOT NULL,
    elevation REAL,
    hand      REAL,
    slope     REAL,
    twi       REAL,
    built_up  REAL,
    log_upa   REAL,                        -- flow accumulation (drainage model)
    UNIQUE (grid_x, grid_y)
);
CREATE INDEX IF NOT EXISTS idx_grid_xy ON topographic_grid(grid_x, grid_y);

CREATE TABLE IF NOT EXISTS permanent_water (
    water_id  INTEGER PRIMARY KEY,
    latitude  REAL,
    longitude REAL,
    area_m2   REAL
);

CREATE TABLE IF NOT EXISTS critical_infrastructure (
    infra_id  INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    type      TEXT,
    latitude  REAL NOT NULL,
    longitude REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settlements (
    settlement_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    latitude      REAL NOT NULL,
    longitude     REAL NOT NULL,
    grid_row      INTEGER,
    grid_col      INTEGER
);

-- ------------------------------------------------ model and dataset lineage --
-- Chapter 4 requires a displayed prediction to be traceable to the exact label
-- parameters that defined the training target (DR-06, DR-08).
CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version_id INTEGER PRIMARY KEY,
    artefact_path      TEXT NOT NULL,
    susceptibility     TEXT,               -- 'terrain' | 'drainage'
    mode               TEXT,               -- 'forecast' | 'nwp'
    n_samples          INTEGER,
    n_seasons          INTEGER,
    positive_rate      REAL,
    built_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS label_parameters (
    parameter_id       INTEGER PRIMARY KEY,
    dataset_version_id INTEGER NOT NULL
                       REFERENCES dataset_versions(dataset_version_id) ON DELETE CASCADE,
    parameter_name     TEXT NOT NULL,
    parameter_value    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id   INTEGER PRIMARY KEY,
    dataset_version_id INTEGER REFERENCES dataset_versions(dataset_version_id),
    artefact_path      TEXT NOT NULL,
    architecture       TEXT,
    test_f1            REAL,
    test_iou           REAL,
    test_precision     REAL,
    test_recall        REAL,
    is_active          INTEGER NOT NULL DEFAULT 0,   -- exactly one active (NFR-07)
    trained_at         TEXT NOT NULL
);

-- ------------------------------------------------------------- simulations --
CREATE TABLE IF NOT EXISTS simulations (
    sim_id               INTEGER PRIMARY KEY,
    user_id              INTEGER REFERENCES users(user_id),
    model_version_id     INTEGER REFERENCES model_versions(model_version_id),
    created_at           TEXT NOT NULL,
    rainfall_scenario_mm REAL NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending','running','complete','failed')),
    exec_time_sec        REAL
);
CREATE INDEX IF NOT EXISTS idx_sim_user ON simulations(user_id);
CREATE INDEX IF NOT EXISTS idx_sim_created ON simulations(created_at);

CREATE TABLE IF NOT EXISTS simulation_results (
    result_id        INTEGER PRIMARY KEY,
    sim_id           INTEGER NOT NULL REFERENCES simulations(sim_id) ON DELETE CASCADE,
    computed_at      TEXT NOT NULL,
    avg_probability  REAL,
    pixels_flooded   INTEGER,
    flooded_fraction REAL,
    result_path      TEXT                  -- array on disk, not inlined
);

CREATE TABLE IF NOT EXISTS flood_pixels (
    pixel_id          INTEGER PRIMARY KEY,
    result_id         INTEGER NOT NULL REFERENCES simulation_results(result_id) ON DELETE CASCADE,
    grid_x            INTEGER NOT NULL,
    grid_y            INTEGER NOT NULL,
    flood_probability REAL NOT NULL,
    hand_value        REAL,
    elevation         REAL,
    built_up          REAL
);
CREATE INDEX IF NOT EXISTS idx_flood_pixels_result ON flood_pixels(result_id);

CREATE TABLE IF NOT EXISTS affected_settlements (
    sim_id                    INTEGER NOT NULL REFERENCES simulations(sim_id) ON DELETE CASCADE,
    settlement_id             INTEGER NOT NULL REFERENCES settlements(settlement_id),
    susceptibility_percentile REAL,
    PRIMARY KEY (sim_id, settlement_id)
);

CREATE TABLE IF NOT EXISTS prediction_archive (
    archive_id       INTEGER PRIMARY KEY,
    sim_id           INTEGER REFERENCES simulations(sim_id) ON DELETE CASCADE,
    archived_at      TEXT NOT NULL,
    storage_location TEXT NOT NULL,
    compressed_size  INTEGER
);

-- ------------------------------------------------------------------ audit --
-- Append-only (FR-15, DR-09). No UPDATE or DELETE is issued against this table.
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id         INTEGER PRIMARY KEY,
    occurred_at      TEXT NOT NULL,
    user_id          INTEGER REFERENCES users(user_id),
    action           TEXT NOT NULL,
    model_version_id INTEGER REFERENCES model_versions(model_version_id),
    detail           TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(occurred_at);

CREATE TABLE IF NOT EXISTS alert_subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    sub_county      TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    UNIQUE (user_id, sub_county)
);
