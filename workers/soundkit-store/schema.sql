CREATE TABLE IF NOT EXISTS objects (
  object_id TEXT PRIMARY KEY,
  status TEXT NOT NULL CHECK (status IN ('uploading', 'sealed', 'aborted', 'failed')),
  codec TEXT NOT NULL,
  content_type TEXT NOT NULL,
  timescale INTEGER NOT NULL,
  opus_frame_ms REAL,
  target_chunk_bytes INTEGER NOT NULL,
  max_chunk_bytes INTEGER NOT NULL,
  committed_bytes INTEGER NOT NULL DEFAULT 0,
  duration_frames INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  frame_count INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  sealed_at INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
  object_id TEXT NOT NULL,
  chunk_no INTEGER NOT NULL,
  start_offset INTEGER NOT NULL,
  byte_len INTEGER NOT NULL,
  r2_key TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('committed', 'failed')),
  crc32 INTEGER,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (object_id, chunk_no),
  FOREIGN KEY (object_id) REFERENCES objects(object_id)
);

CREATE INDEX IF NOT EXISTS chunks_range
  ON chunks(object_id, status, start_offset);

CREATE TABLE IF NOT EXISTS frame_index (
  object_id TEXT NOT NULL,
  entry_no INTEGER NOT NULL,
  chunk_no INTEGER NOT NULL,
  byte_offset INTEGER NOT NULL,
  start_frame INTEGER NOT NULL,
  frame_count INTEGER NOT NULL,
  PRIMARY KEY (object_id, entry_no),
  FOREIGN KEY (object_id, chunk_no) REFERENCES chunks(object_id, chunk_no)
);

CREATE INDEX IF NOT EXISTS frame_seek
  ON frame_index(object_id, start_frame);
