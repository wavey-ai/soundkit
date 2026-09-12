#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use serde::{Deserialize, Serialize};

const DEFAULT_TARGET_CHUNK_BYTES: i64 = 256 * 1024;
const DEFAULT_MAX_CHUNK_BYTES: i64 = 512 * 1024;
const DEFAULT_MAX_RANGE_RESPONSE_BYTES: i64 = 4 * 1024 * 1024;
const MAX_REQUEST_BODY_BYTES: i64 = 8 * 1024 * 1024;
const SOUNDKIT_FRAME_HEADER_BASE_BYTES: usize = 8;
const SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES: usize = 8;
const SOUNDKIT_SAMPLE_RATES: [u32; 11] = [
    8_000, 12_000, 16_000, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000,
];
const INDEX_MAGIC: [u8; 8] = [0x53, 0x4b, 0x49, 0x44, 0x58, 0x32, 0x00, 0x00];
const INDEX_VERSION: u16 = 1;
const INDEX_ENTRY_BYTES: u16 = 16;
const INDEX_HEADER_BYTES: usize = 32;
const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;
const SCHEMA_SQL: &str = include_str!("../schema.sql");

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    ok: bool,
    service: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateUploadRequest {
    #[serde(default)]
    reset: bool,
    #[serde(default)]
    codec: Option<String>,
    #[serde(default)]
    content_type: Option<String>,
    #[serde(default)]
    timescale: Option<i64>,
    #[serde(default)]
    opus_frame_ms: Option<f64>,
    #[serde(default)]
    target_chunk_bytes: Option<i64>,
    #[serde(default)]
    max_chunk_bytes: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ObjectRow {
    object_id: String,
    status: String,
    codec: String,
    content_type: String,
    timescale: i64,
    opus_frame_ms: Option<f64>,
    target_chunk_bytes: i64,
    max_chunk_bytes: i64,
    committed_bytes: i64,
    duration_frames: i64,
    chunk_count: i64,
    frame_count: i64,
    updated_at: i64,
    sealed_at: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExistingObjectRow {
    status: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChunkRow {
    start_offset: i64,
    byte_len: i64,
    r2_key: String,
}

#[derive(Debug, Clone, Deserialize)]
struct FrameIndexRow {
    byte_offset: i64,
    start_frame: i64,
}

#[derive(Debug, Clone, Deserialize)]
struct FrameSeekRow {
    byte_offset: i64,
    start_frame: i64,
    frame_count: i64,
}

#[derive(Debug, Clone, Deserialize)]
struct TailFrameRow {
    start_frame: i64,
    frame_count: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ManifestResponse {
    object_id: String,
    status: String,
    codec: String,
    content_type: String,
    timescale: i64,
    opus_frame_ms: Option<f64>,
    target_chunk_bytes: i64,
    max_chunk_bytes: i64,
    committed_bytes: i64,
    duration_frames: i64,
    chunk_count: i64,
    frame_count: i64,
    storage_layout: &'static str,
    chunks_prefix: String,
    index_key: String,
    manifest_key: String,
    sealed_at: Option<i64>,
    updated_at: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AppendChunkResponse {
    object_id: String,
    chunk_no: i64,
    r2_key: String,
    start_offset: i64,
    byte_length: i64,
    frame_count: i64,
    committed_bytes: i64,
    duration_frames: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SeekResponse {
    object_id: String,
    query_frame: i64,
    byte_offset: i64,
    start_frame: i64,
    frame_count: i64,
    timescale: i64,
}

#[derive(Debug, Clone, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Clone)]
struct HttpError {
    status: u16,
    message: String,
}

impl HttpError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct ScannedFrame {
    byte_offset: i64,
    start_frame: i64,
    frame_count: i64,
}

#[derive(Debug, Clone)]
struct DecodedSoundKitFrameHeader {
    payload_size: usize,
    frame_count: i64,
    pts: Option<i64>,
    header_bytes: usize,
}

#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use worker::*;

#[cfg(target_arch = "wasm32")]
#[event(fetch)]
pub async fn main(
    mut request: Request,
    env: worker::Env,
    _ctx: worker::Context,
) -> Result<Response> {
    console_error_panic_hook::set_once();

    let origin = request.headers().get("Origin").ok().flatten();
    if request.method() == Method::Options {
        return empty_response(204, origin.as_deref());
    }

    match handle_request(&mut request, &env, origin.as_deref()).await {
        Ok(response) => Ok(response),
        Err(error) => {
            console_error!(
                "{}",
                serde_json::json!({
                    "event": "soundkit-store-error",
                    "status": error.status,
                    "message": error.message
                })
            );
            json_response(
                &ErrorResponse {
                    error: error.message,
                },
                error.status,
                origin.as_deref(),
            )
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn handle_request(
    request: &mut Request,
    env: &worker::Env,
    origin: Option<&str>,
) -> std::result::Result<Response, HttpError> {
    let url = request.url().map_err(internal_error)?;
    if url.path() == "/health" {
        return json_response(
            &HealthResponse {
                ok: true,
                service: "soundkit-store",
            },
            200,
            origin,
        )
        .map_err(internal_error);
    }

    let route = route_request(url.path())?;
    let db = env.d1("SOUNDKIT_DB").map_err(internal_error)?;
    ensure_schema(&db).await.map_err(internal_error)?;

    match (request.method(), route.action.as_str()) {
        (Method::Post, "uploads") => {
            let options = read_create_upload_request(request).await?;
            let manifest = create_upload(&db, env, &route.object_id, options).await?;
            json_response(&manifest, 200, origin).map_err(internal_error)
        }
        (Method::Post, "chunks") => {
            let bytes = read_bounded_body(request, MAX_REQUEST_BODY_BYTES).await?;
            let result = append_chunk(&db, env, &route.object_id, bytes).await?;
            json_response(&result, 200, origin).map_err(internal_error)
        }
        (Method::Post, "seal") => {
            let manifest = seal_object(&db, env, &route.object_id).await?;
            json_response(&manifest, 200, origin).map_err(internal_error)
        }
        (Method::Get, "manifest") => {
            let object = get_object(&db, &route.object_id).await?;
            json_response(&manifest_from_object(&object), 200, origin).map_err(internal_error)
        }
        (Method::Get, "index") => {
            let bytes = get_index_bytes(&db, &route.object_id).await?;
            binary_response(bytes, 200, "application/octet-stream", "no-cache", origin)
                .map_err(internal_error)
        }
        (Method::Get, "stream") => {
            let range = parse_range_header(request)?;
            let stream = get_range_bytes(&db, env, &route.object_id, range).await?;
            stream_response(
                stream,
                request.headers().get("Range").ok().flatten().is_some(),
                origin,
            )
            .map_err(internal_error)
        }
        (Method::Get, "seek") => {
            let seek = seek_object(&db, &route.object_id, &url).await?;
            json_response(&seek, 200, origin).map_err(internal_error)
        }
        _ => Err(HttpError::new(405, "method not allowed")),
    }
}

struct Route {
    object_id: String,
    action: String,
}

fn route_request(pathname: &str) -> std::result::Result<Route, HttpError> {
    let parts = pathname
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() != 4 || parts[0] != "v1" || parts[1] != "objects" {
        return Err(HttpError::new(404, "not found"));
    }
    let object_id = decode_uri_component(parts[2])?;
    validate_object_id(&object_id)?;
    Ok(Route {
        object_id,
        action: parts[3].to_string(),
    })
}

fn validate_object_id(object_id: &str) -> std::result::Result<(), HttpError> {
    if object_id.is_empty() || object_id.len() > 180 {
        return Err(HttpError::new(
            400,
            "objectId must be 1-180 URL-safe characters",
        ));
    }
    if !object_id
        .bytes()
        .all(|byte| matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'~' | b':' | b'-'))
    {
        return Err(HttpError::new(
            400,
            "objectId must contain only URL-safe characters",
        ));
    }
    Ok(())
}

fn decode_uri_component(value: &str) -> std::result::Result<String, HttpError> {
    urlencoding::decode(value)
        .map(|decoded| decoded.into_owned())
        .map_err(|_| HttpError::new(400, "invalid URL encoding"))
}

#[cfg(target_arch = "wasm32")]
async fn ensure_schema(db: &worker::d1::D1Database) -> Result<()> {
    for statement in SCHEMA_SQL.split(';') {
        let statement = statement.trim();
        if !statement.is_empty() {
            db.prepare(statement).run().await?;
        }
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn read_create_upload_request(
    request: &mut Request,
) -> std::result::Result<CreateUploadRequest, HttpError> {
    let content_type = request
        .headers()
        .get("Content-Type")
        .map_err(internal_http_error)?
        .unwrap_or_default()
        .to_ascii_lowercase();
    let media_type = content_type.split(';').next().unwrap_or("").trim();
    if media_type.is_empty() {
        return Ok(CreateUploadRequest {
            reset: false,
            codec: None,
            content_type: None,
            timescale: None,
            opus_frame_ms: None,
            target_chunk_bytes: None,
            max_chunk_bytes: None,
        });
    }
    if media_type != "application/json" {
        return Err(HttpError::new(415, "expected application/json"));
    }
    request.json().await.map_err(internal_http_error)
}

#[cfg(target_arch = "wasm32")]
async fn read_bounded_body(
    request: &mut Request,
    max_bytes: i64,
) -> std::result::Result<Vec<u8>, HttpError> {
    let declared = request
        .headers()
        .get("Content-Length")
        .map_err(internal_http_error)?
        .and_then(|value| value.parse::<i64>().ok());
    if declared.is_some_and(|length| length > max_bytes) {
        return Err(HttpError::new(
            413,
            format!("request body exceeds {max_bytes} bytes"),
        ));
    }
    let bytes = request.bytes().await.map_err(internal_http_error)?;
    if bytes.len() as i64 > max_bytes {
        return Err(HttpError::new(
            413,
            format!("request body exceeds {max_bytes} bytes"),
        ));
    }
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
async fn create_upload(
    db: &worker::d1::D1Database,
    env: &worker::Env,
    object_id: &str,
    options: CreateUploadRequest,
) -> std::result::Result<ManifestResponse, HttpError> {
    let default_max = env_integer(env, "DEFAULT_MAX_CHUNK_BYTES", DEFAULT_MAX_CHUNK_BYTES);
    let target_chunk_bytes = bounded_i64(
        options.target_chunk_bytes,
        env_integer(
            env,
            "DEFAULT_TARGET_CHUNK_BYTES",
            DEFAULT_TARGET_CHUNK_BYTES,
        ),
        4096,
        default_max,
        "targetChunkBytes",
    )?;
    let max_chunk_bytes = bounded_i64(
        options.max_chunk_bytes,
        default_max,
        target_chunk_bytes,
        MAX_REQUEST_BODY_BYTES,
        "maxChunkBytes",
    )?;
    let timescale = bounded_i64(options.timescale, 48_000, 1, 384_000, "timescale")?;
    let opus_frame_ms = options.opus_frame_ms.unwrap_or(20.0);
    if !opus_frame_ms.is_finite() || opus_frame_ms <= 0.0 {
        return Err(HttpError::new(400, "opusFrameMs must be a positive number"));
    }
    let codec = options.codec.unwrap_or_else(|| "opus".to_string());
    let content_type = options
        .content_type
        .unwrap_or_else(|| "audio/soundkit".to_string());
    let now = now_ms()?;

    if let Some(existing) = get_existing_object(db, object_id).await? {
        if !options.reset {
            return Err(HttpError::new(
                409,
                format!(
                    "object {object_id} already exists with status {}",
                    existing.status
                ),
            ));
        }
        delete_object_rows(db, object_id).await?;
    }

    bind_run(
        db,
        "INSERT INTO objects (
             object_id, status, codec, content_type, timescale, opus_frame_ms,
             target_chunk_bytes, max_chunk_bytes, committed_bytes, duration_frames,
             chunk_count, frame_count, created_at, updated_at
         ) VALUES (?, 'uploading', ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, ?, ?)",
        vec![
            JsValue::from_str(object_id),
            JsValue::from_str(&codec),
            JsValue::from_str(&content_type),
            js_i64(timescale)?,
            JsValue::from_f64(opus_frame_ms),
            js_i64(target_chunk_bytes)?,
            js_i64(max_chunk_bytes)?,
            js_i64(now)?,
            js_i64(now)?,
        ],
    )
    .await?;

    let object = get_object(db, object_id).await?;
    Ok(manifest_from_object(&object))
}

#[cfg(target_arch = "wasm32")]
async fn append_chunk(
    db: &worker::d1::D1Database,
    env: &worker::Env,
    object_id: &str,
    bytes: Vec<u8>,
) -> std::result::Result<AppendChunkResponse, HttpError> {
    let object = get_object(db, object_id).await?;
    if object.status != "uploading" {
        return Err(HttpError::new(
            409,
            format!("object {object_id} is {}", object.status),
        ));
    }
    if bytes.is_empty() {
        return Err(HttpError::new(400, "chunk must not be empty"));
    }
    if bytes.len() as i64 > object.max_chunk_bytes {
        return Err(HttpError::new(
            413,
            format!("chunk exceeds maxChunkBytes {}", object.max_chunk_bytes),
        ));
    }

    let next_frame_start = next_frame_start(db, object_id).await?;
    let frames = scan_soundkit_frames(&bytes, object.committed_bytes, next_frame_start)?;
    if frames.is_empty() {
        return Err(HttpError::new(
            400,
            "chunk contains no complete SoundKit frames",
        ));
    }

    let chunk_no = object.chunk_count;
    let start_offset = object.committed_bytes;
    let r2_key = chunk_key(object_id, chunk_no);
    let bucket = env.bucket("SOUNDKIT_BUCKET").map_err(internal_http_error)?;
    let metadata = worker::HttpMetadata {
        content_type: Some("audio/soundkit".to_string()),
        cache_control: Some("no-store".to_string()),
        ..Default::default()
    };
    let custom_metadata = HashMap::from([
        ("objectId".to_string(), object_id.to_string()),
        ("chunkNo".to_string(), chunk_no.to_string()),
        ("startOffset".to_string(), start_offset.to_string()),
        ("byteLength".to_string(), bytes.len().to_string()),
    ]);
    bucket
        .put(&r2_key, bytes.clone())
        .http_metadata(metadata)
        .custom_metadata(custom_metadata)
        .execute()
        .await
        .map_err(internal_http_error)?;

    let now = now_ms()?;
    let first_entry = object.frame_count;
    let duration_frames = frames
        .iter()
        .map(|frame| frame.start_frame + frame.frame_count)
        .max()
        .unwrap_or(object.duration_frames)
        .max(object.duration_frames);
    let committed_bytes = object.committed_bytes + bytes.len() as i64;

    let mut statements = Vec::with_capacity(frames.len() + 2);
    statements.push(
        db.prepare(
            "INSERT INTO chunks (
                 object_id, chunk_no, start_offset, byte_len, r2_key, status, created_at
             ) VALUES (?, ?, ?, ?, ?, 'committed', ?)",
        )
        .bind(&[
            JsValue::from_str(object_id),
            js_i64(chunk_no)?,
            js_i64(start_offset)?,
            js_i64(bytes.len() as i64)?,
            JsValue::from_str(&r2_key),
            js_i64(now)?,
        ])
        .map_err(internal_http_error)?,
    );

    for (index, frame) in frames.iter().enumerate() {
        statements.push(
            db.prepare(
                "INSERT INTO frame_index (
                     object_id, entry_no, chunk_no, byte_offset, start_frame, frame_count
                 ) VALUES (?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                JsValue::from_str(object_id),
                js_i64(first_entry + index as i64)?,
                js_i64(chunk_no)?,
                js_i64(frame.byte_offset)?,
                js_i64(frame.start_frame)?,
                js_i64(frame.frame_count)?,
            ])
            .map_err(internal_http_error)?,
        );
    }

    statements.push(
        db.prepare(
            "UPDATE objects
             SET committed_bytes = ?,
                 duration_frames = ?,
                 chunk_count = chunk_count + 1,
                 frame_count = frame_count + ?,
                 updated_at = ?
             WHERE object_id = ?
               AND status = 'uploading'
               AND chunk_count = ?",
        )
        .bind(&[
            js_i64(committed_bytes)?,
            js_i64(duration_frames)?,
            js_i64(frames.len() as i64)?,
            js_i64(now)?,
            JsValue::from_str(object_id),
            js_i64(chunk_no)?,
        ])
        .map_err(internal_http_error)?,
    );

    db.batch(statements).await.map_err(internal_http_error)?;

    Ok(AppendChunkResponse {
        object_id: object_id.to_string(),
        chunk_no,
        r2_key,
        start_offset,
        byte_length: bytes.len() as i64,
        frame_count: frames.len() as i64,
        committed_bytes,
        duration_frames,
    })
}

#[cfg(target_arch = "wasm32")]
async fn get_range_bytes(
    db: &worker::d1::D1Database,
    env: &worker::Env,
    object_id: &str,
    range: Option<(i64, Option<i64>)>,
) -> std::result::Result<RangeBytes, HttpError> {
    let object = get_object(db, object_id).await?;
    if object.committed_bytes <= 0 {
        return Err(HttpError::new(416, "object has no committed bytes"));
    }
    let (start, requested_end) = range.unwrap_or((0, None));
    if start < 0 || start >= object.committed_bytes {
        return Err(HttpError::new(
            416,
            "requested range is outside committed bytes",
        ));
    }
    let end = requested_end
        .unwrap_or(object.committed_bytes - 1)
        .min(object.committed_bytes - 1);
    if end < start {
        return Err(HttpError::new(416, "invalid byte range"));
    }
    let response_bytes = end - start + 1;
    let max_range = env_integer(
        env,
        "MAX_RANGE_RESPONSE_BYTES",
        DEFAULT_MAX_RANGE_RESPONSE_BYTES,
    );
    if response_bytes > max_range {
        return Err(HttpError::new(
            416,
            format!("range exceeds MAX_RANGE_RESPONSE_BYTES {max_range}"),
        ));
    }

    let rows = select_chunks_for_range(db, object_id, start, end).await?;
    if rows.is_empty() {
        return Err(HttpError::new(416, "requested range is not committed"));
    }

    let bucket = env.bucket("SOUNDKIT_BUCKET").map_err(internal_http_error)?;
    let mut body = Vec::with_capacity(response_bytes as usize);
    for row in rows {
        let copy_start = start.max(row.start_offset);
        let copy_end = (end + 1).min(row.start_offset + row.byte_len);
        let offset = (copy_start - row.start_offset) as u64;
        let length = (copy_end - copy_start) as u64;
        let Some(object) = bucket
            .get(&row.r2_key)
            .range(worker::Range::OffsetWithLength { offset, length })
            .execute()
            .await
            .map_err(internal_http_error)?
        else {
            return Err(HttpError::new(
                500,
                format!("missing committed R2 chunk {}", row.r2_key),
            ));
        };
        let Some(chunk_body) = object.body() else {
            return Err(HttpError::new(
                500,
                format!("missing R2 chunk body {}", row.r2_key),
            ));
        };
        body.extend(chunk_body.bytes().await.map_err(internal_http_error)?);
    }

    Ok(RangeBytes {
        body,
        start,
        end,
        total: object.committed_bytes,
        content_type: object.content_type,
    })
}

struct RangeBytes {
    body: Vec<u8>,
    start: i64,
    end: i64,
    total: i64,
    content_type: String,
}

#[cfg(target_arch = "wasm32")]
async fn get_index_bytes(
    db: &worker::d1::D1Database,
    object_id: &str,
) -> std::result::Result<Vec<u8>, HttpError> {
    let object = get_object(db, object_id).await?;
    let rows = select_frame_index(db, object_id).await?;
    Ok(encode_sidecar_index(
        object.timescale as u32,
        object.duration_frames as u64,
        rows.iter()
            .map(|row| (row.byte_offset as u64, row.start_frame as u64))
            .collect::<Vec<_>>()
            .as_slice(),
    ))
}

#[cfg(target_arch = "wasm32")]
async fn seal_object(
    db: &worker::d1::D1Database,
    env: &worker::Env,
    object_id: &str,
) -> std::result::Result<ManifestResponse, HttpError> {
    let object = get_object(db, object_id).await?;
    if matches!(object.status.as_str(), "aborted" | "failed") {
        return Err(HttpError::new(
            409,
            format!("object {object_id} is {}", object.status),
        ));
    }
    if object.committed_bytes <= 0 {
        return Err(HttpError::new(409, "cannot seal an empty object"));
    }

    let index_bytes = get_index_bytes(db, object_id).await?;
    let now = now_ms()?;
    bind_run(
        db,
        "UPDATE objects
         SET status = 'sealed', sealed_at = ?, updated_at = ?
         WHERE object_id = ?",
        vec![js_i64(now)?, js_i64(now)?, JsValue::from_str(object_id)],
    )
    .await?;

    let object = get_object(db, object_id).await?;
    let manifest = manifest_from_object(&object);
    let bucket = env.bucket("SOUNDKIT_BUCKET").map_err(internal_http_error)?;
    let final_cache_control = env_string(env, "FINAL_STREAM_CACHE_CONTROL")
        .unwrap_or_else(|| "public, max-age=31536000, immutable".to_string());
    bucket
        .put(index_key(object_id), index_bytes)
        .http_metadata(worker::HttpMetadata {
            content_type: Some("application/octet-stream".to_string()),
            cache_control: Some(final_cache_control),
            ..Default::default()
        })
        .execute()
        .await
        .map_err(internal_http_error)?;
    bucket
        .put(
            manifest_key(object_id),
            serde_json::to_string_pretty(&manifest).map_err(internal_http_error)?,
        )
        .http_metadata(worker::HttpMetadata {
            content_type: Some("application/json".to_string()),
            cache_control: Some("no-cache".to_string()),
            ..Default::default()
        })
        .execute()
        .await
        .map_err(internal_http_error)?;
    Ok(manifest)
}

#[cfg(target_arch = "wasm32")]
async fn seek_object(
    db: &worker::d1::D1Database,
    object_id: &str,
    url: &url::Url,
) -> std::result::Result<SeekResponse, HttpError> {
    let object = get_object(db, object_id).await?;
    let mut query_frame = None;
    for (name, value) in url.query_pairs() {
        match name.as_ref() {
            "frame" => {
                query_frame = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| HttpError::new(400, "frame must be an integer"))?,
                );
            }
            "timeMs" => {
                let time_ms = value
                    .parse::<f64>()
                    .map_err(|_| HttpError::new(400, "timeMs must be a number"))?;
                if !time_ms.is_finite() || time_ms < 0.0 {
                    return Err(HttpError::new(400, "timeMs must be a non-negative number"));
                }
                query_frame = Some(((time_ms / 1000.0) * object.timescale as f64).floor() as i64);
            }
            _ => {}
        }
    }
    let query_frame =
        query_frame.ok_or_else(|| HttpError::new(400, "expected frame or timeMs query"))?;
    if query_frame < 0 {
        return Err(HttpError::new(400, "seek frame must be non-negative"));
    }

    let args = vec![JsValue::from_str(object_id), js_i64(query_frame)?];
    let row = db
        .prepare(
            "SELECT byte_offset, start_frame, frame_count
             FROM frame_index
             WHERE object_id = ? AND start_frame <= ?
             ORDER BY start_frame DESC
             LIMIT 1",
        )
        .bind(&args)
        .map_err(internal_http_error)?
        .first::<FrameSeekRow>(None)
        .await
        .map_err(internal_http_error)?
        .ok_or_else(|| HttpError::new(404, "no indexed frame at or before seek point"))?;

    Ok(SeekResponse {
        object_id: object_id.to_string(),
        query_frame,
        byte_offset: row.byte_offset,
        start_frame: row.start_frame,
        frame_count: row.frame_count,
        timescale: object.timescale,
    })
}

#[cfg(target_arch = "wasm32")]
async fn get_existing_object(
    db: &worker::d1::D1Database,
    object_id: &str,
) -> std::result::Result<Option<ExistingObjectRow>, HttpError> {
    let args = [JsValue::from_str(object_id)];
    db.prepare("SELECT status FROM objects WHERE object_id = ? LIMIT 1")
        .bind(&args)
        .map_err(internal_http_error)?
        .first::<ExistingObjectRow>(None)
        .await
        .map_err(internal_http_error)
}

#[cfg(target_arch = "wasm32")]
async fn get_object(
    db: &worker::d1::D1Database,
    object_id: &str,
) -> std::result::Result<ObjectRow, HttpError> {
    let args = [JsValue::from_str(object_id)];
    db.prepare(
        "SELECT object_id, status, codec, content_type, timescale, opus_frame_ms,
                target_chunk_bytes, max_chunk_bytes, committed_bytes, duration_frames,
                chunk_count, frame_count, updated_at, sealed_at
         FROM objects
         WHERE object_id = ?
         LIMIT 1",
    )
    .bind(&args)
    .map_err(internal_http_error)?
    .first::<ObjectRow>(None)
    .await
    .map_err(internal_http_error)?
    .ok_or_else(|| HttpError::new(404, format!("object {object_id} was not found")))
}

#[cfg(target_arch = "wasm32")]
async fn delete_object_rows(
    db: &worker::d1::D1Database,
    object_id: &str,
) -> std::result::Result<(), HttpError> {
    let statements = vec![
        db.prepare("DELETE FROM frame_index WHERE object_id = ?")
            .bind(&[JsValue::from_str(object_id)])
            .map_err(internal_http_error)?,
        db.prepare("DELETE FROM chunks WHERE object_id = ?")
            .bind(&[JsValue::from_str(object_id)])
            .map_err(internal_http_error)?,
        db.prepare("DELETE FROM objects WHERE object_id = ?")
            .bind(&[JsValue::from_str(object_id)])
            .map_err(internal_http_error)?,
    ];
    db.batch(statements).await.map_err(internal_http_error)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn next_frame_start(
    db: &worker::d1::D1Database,
    object_id: &str,
) -> std::result::Result<i64, HttpError> {
    let args = [JsValue::from_str(object_id)];
    let row = db
        .prepare(
            "SELECT start_frame, frame_count
             FROM frame_index
             WHERE object_id = ?
             ORDER BY entry_no DESC
             LIMIT 1",
        )
        .bind(&args)
        .map_err(internal_http_error)?
        .first::<TailFrameRow>(None)
        .await
        .map_err(internal_http_error)?;
    Ok(row
        .map(|row| row.start_frame + row.frame_count)
        .unwrap_or(0))
}

#[cfg(target_arch = "wasm32")]
async fn select_chunks_for_range(
    db: &worker::d1::D1Database,
    object_id: &str,
    start: i64,
    end: i64,
) -> std::result::Result<Vec<ChunkRow>, HttpError> {
    db.prepare(
        "SELECT start_offset, byte_len, r2_key
         FROM chunks
         WHERE object_id = ?
           AND status = 'committed'
           AND start_offset <= ?
           AND start_offset + byte_len > ?
         ORDER BY chunk_no",
    )
    .bind(&[JsValue::from_str(object_id), js_i64(end)?, js_i64(start)?])
    .map_err(internal_http_error)?
    .all()
    .await
    .map_err(internal_http_error)?
    .results::<ChunkRow>()
    .map_err(internal_http_error)
}

#[cfg(target_arch = "wasm32")]
async fn select_frame_index(
    db: &worker::d1::D1Database,
    object_id: &str,
) -> std::result::Result<Vec<FrameIndexRow>, HttpError> {
    db.prepare(
        "SELECT byte_offset, start_frame
         FROM frame_index
         WHERE object_id = ?
         ORDER BY entry_no",
    )
    .bind(&[JsValue::from_str(object_id)])
    .map_err(internal_http_error)?
    .all()
    .await
    .map_err(internal_http_error)?
    .results::<FrameIndexRow>()
    .map_err(internal_http_error)
}

#[cfg(target_arch = "wasm32")]
async fn bind_run(
    db: &worker::d1::D1Database,
    statement: &str,
    values: Vec<JsValue>,
) -> std::result::Result<(), HttpError> {
    db.prepare(statement)
        .bind(&values)
        .map_err(internal_http_error)?
        .run()
        .await
        .map_err(internal_http_error)?;
    Ok(())
}

fn manifest_from_object(object: &ObjectRow) -> ManifestResponse {
    ManifestResponse {
        object_id: object.object_id.clone(),
        status: object.status.clone(),
        codec: object.codec.clone(),
        content_type: object.content_type.clone(),
        timescale: object.timescale,
        opus_frame_ms: object.opus_frame_ms,
        target_chunk_bytes: object.target_chunk_bytes,
        max_chunk_bytes: object.max_chunk_bytes,
        committed_bytes: object.committed_bytes,
        duration_frames: object.duration_frames,
        chunk_count: object.chunk_count,
        frame_count: object.frame_count,
        storage_layout: "r2-chunks-d1-index",
        chunks_prefix: format!("objects/{}/chunks/", object.object_id),
        index_key: index_key(&object.object_id),
        manifest_key: manifest_key(&object.object_id),
        sealed_at: object.sealed_at,
        updated_at: object.updated_at,
    }
}

fn scan_soundkit_frames(
    bytes: &[u8],
    base_byte_offset: i64,
    fallback_start_frame: i64,
) -> std::result::Result<Vec<ScannedFrame>, HttpError> {
    if bytes.is_empty() {
        return Err(HttpError::new(400, "SoundKit packet stream is empty"));
    }
    let mut frames = Vec::new();
    let mut offset = 0usize;
    let mut next_start_frame = fallback_start_frame;

    while offset < bytes.len() {
        let header = decode_soundkit_frame_header(bytes, offset)?;
        let frame_end = offset
            .checked_add(header.header_bytes)
            .and_then(|value| value.checked_add(header.payload_size))
            .ok_or_else(|| HttpError::new(400, "SoundKit frame length overflow"))?;
        if frame_end > bytes.len() {
            return Err(HttpError::new(400, "chunk ends inside a SoundKit frame"));
        }
        let start_frame = header.pts.unwrap_or(next_start_frame);
        frames.push(ScannedFrame {
            byte_offset: base_byte_offset + offset as i64,
            start_frame,
            frame_count: header.frame_count,
        });
        next_start_frame = start_frame + header.frame_count;
        offset = frame_end;
    }

    Ok(frames)
}

fn decode_soundkit_frame_header(
    bytes: &[u8],
    start_offset: usize,
) -> std::result::Result<DecodedSoundKitFrameHeader, HttpError> {
    assert_available(bytes, start_offset, SOUNDKIT_FRAME_HEADER_BASE_BYTES)?;
    let word = read_u32_be(bytes, start_offset)?;
    let size_word = read_u32_be(bytes, start_offset + 4)?;
    let mut offset = start_offset + SOUNDKIT_FRAME_HEADER_BASE_BYTES;

    let magic = (word >> 26) & 0x3f;
    if magic != 0x2b {
        return Err(HttpError::new(
            400,
            format!("invalid SoundKit v2 frame magic 0x{magic:x}"),
        ));
    }
    let version = (word >> 24) & 0x3;
    if version != 2 {
        return Err(HttpError::new(
            400,
            format!("unsupported SoundKit frame version {version}"),
        ));
    }

    let flags = ((word >> 16) & 0xff) as u8;
    let encoding = (word >> 12) & 0xf;
    if encoding != 2 {
        return Err(HttpError::new(400, "SoundKit frame is not Opus"));
    }
    let sample_rate_index = ((word >> 8) & 0xf) as usize;
    if SOUNDKIT_SAMPLE_RATES.get(sample_rate_index).is_none() {
        return Err(HttpError::new(400, "invalid SoundKit sample-rate code"));
    }
    let bits_index = word & 0x7;
    if bits_index > 5 {
        return Err(HttpError::new(400, "invalid SoundKit bits code"));
    }
    if flags & 0x02 != 0 && flags & 0x01 == 0 {
        return Err(HttpError::new(
            400,
            "SoundKit frame has u64 id flag without id flag",
        ));
    }

    let (payload_size, frame_count) = if flags & 0x20 != 0 {
        if size_word != 0xffff_ffff {
            return Err(HttpError::new(
                400,
                "extended SoundKit sizes must use short-size sentinels",
            ));
        }
        assert_available(bytes, offset, SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES)?;
        let payload_size = read_u32_be(bytes, offset)? as usize;
        let frame_count = read_u32_be(bytes, offset + 4)? as i64;
        offset += SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES;
        (payload_size, frame_count)
    } else {
        let payload_size = ((size_word >> 16) & 0xffff) as usize;
        let frame_count = (size_word & 0xffff) as i64;
        if payload_size == 0xffff || frame_count == 0xffff {
            return Err(HttpError::new(
                400,
                "short SoundKit size sentinel requires extended sizes",
            ));
        }
        (payload_size, frame_count)
    };

    if payload_size == 0 || frame_count <= 0 {
        return Err(HttpError::new(400, "SoundKit frame size is invalid"));
    }

    if flags & 0x01 != 0 {
        offset += if flags & 0x02 != 0 { 8 } else { 4 };
        assert_available(bytes, offset, 0)?;
    }

    let pts = if flags & 0x04 != 0 {
        assert_available(bytes, offset, 8)?;
        let pts = read_u64_be(bytes, offset)?;
        if pts > MAX_SAFE_INTEGER as u64 {
            return Err(HttpError::new(
                400,
                "SoundKit PTS exceeds safe integer range",
            ));
        }
        offset += 8;
        Some(pts as i64)
    } else {
        None
    };

    if flags & 0x08 != 0 {
        offset += 4;
        assert_available(bytes, offset, 0)?;
    }

    Ok(DecodedSoundKitFrameHeader {
        payload_size,
        frame_count,
        pts,
        header_bytes: offset - start_offset,
    })
}

fn encode_sidecar_index(timescale: u32, duration_frames: u64, entries: &[(u64, u64)]) -> Vec<u8> {
    let mut output = vec![0u8; INDEX_HEADER_BYTES + entries.len() * INDEX_ENTRY_BYTES as usize];
    output[0..8].copy_from_slice(&INDEX_MAGIC);
    output[8..10].copy_from_slice(&INDEX_VERSION.to_le_bytes());
    output[10..12].copy_from_slice(&INDEX_ENTRY_BYTES.to_le_bytes());
    output[12..16].copy_from_slice(&timescale.to_le_bytes());
    output[16..24].copy_from_slice(&(entries.len() as u64).to_le_bytes());
    output[24..32].copy_from_slice(&duration_frames.to_le_bytes());

    let mut offset = INDEX_HEADER_BYTES;
    for (byte_offset, start_frame) in entries {
        output[offset..offset + 8].copy_from_slice(&byte_offset.to_le_bytes());
        output[offset + 8..offset + 16].copy_from_slice(&start_frame.to_le_bytes());
        offset += INDEX_ENTRY_BYTES as usize;
    }
    output
}

fn assert_available(
    bytes: &[u8],
    offset: usize,
    length: usize,
) -> std::result::Result<(), HttpError> {
    if offset
        .checked_add(length)
        .is_some_and(|end| end <= bytes.len())
    {
        Ok(())
    } else {
        Err(HttpError::new(400, "incomplete SoundKit frame header"))
    }
}

fn read_u32_be(bytes: &[u8], offset: usize) -> std::result::Result<u32, HttpError> {
    assert_available(bytes, offset, 4)?;
    Ok(u32::from_be_bytes(
        bytes[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_u64_be(bytes: &[u8], offset: usize) -> std::result::Result<u64, HttpError> {
    assert_available(bytes, offset, 8)?;
    Ok(u64::from_be_bytes(
        bytes[offset..offset + 8].try_into().unwrap(),
    ))
}

fn chunk_key(object_id: &str, chunk_no: i64) -> String {
    format!("objects/{object_id}/chunks/{chunk_no:010}.soundkit")
}

fn index_key(object_id: &str) -> String {
    format!("objects/{object_id}/stream.soundkit.idx")
}

fn manifest_key(object_id: &str) -> String {
    format!("objects/{object_id}/manifest.json")
}

fn bounded_i64(
    value: Option<i64>,
    fallback: i64,
    min: i64,
    max: i64,
    label: &str,
) -> std::result::Result<i64, HttpError> {
    let resolved = value.unwrap_or(fallback);
    if resolved < min || resolved > max {
        Err(HttpError::new(
            400,
            format!("{label} must be between {min} and {max}"),
        ))
    } else {
        Ok(resolved)
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_range_header(
    request: &Request,
) -> std::result::Result<Option<(i64, Option<i64>)>, HttpError> {
    let Some(header) = request
        .headers()
        .get("Range")
        .map_err(internal_http_error)?
    else {
        return Ok(None);
    };
    let value = header.trim();
    let Some(rest) = value.strip_prefix("bytes=") else {
        return Err(HttpError::new(
            416,
            "only single explicit byte ranges are supported",
        ));
    };
    let Some((start, end)) = rest.split_once('-') else {
        return Err(HttpError::new(416, "invalid byte range"));
    };
    if start.is_empty() {
        return Err(HttpError::new(416, "suffix byte ranges are not supported"));
    }
    let start = start
        .parse::<i64>()
        .map_err(|_| HttpError::new(416, "invalid byte range"))?;
    let end = if end.is_empty() {
        None
    } else {
        Some(
            end.parse::<i64>()
                .map_err(|_| HttpError::new(416, "invalid byte range"))?,
        )
    };
    if start < 0 || end.is_some_and(|end| end < start) {
        return Err(HttpError::new(416, "invalid byte range"));
    }
    Ok(Some((start, end)))
}

#[cfg(target_arch = "wasm32")]
fn stream_response(
    range: RangeBytes,
    requested_range: bool,
    origin: Option<&str>,
) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Content-Type", &range.content_type)?;
    headers.set("Accept-Ranges", "bytes")?;
    headers.set("Content-Length", &range.body.len().to_string())?;
    headers.set(
        "Content-Range",
        &format!("bytes {}-{}/{}", range.start, range.end, range.total),
    )?;
    headers.set("Cache-Control", "no-store")?;
    with_cors(
        Response::from_bytes(range.body)?
            .with_status(if requested_range { 206 } else { 200 })
            .with_headers(headers),
        origin,
    )
}

#[cfg(target_arch = "wasm32")]
fn binary_response(
    body: Vec<u8>,
    status: u16,
    content_type: &str,
    cache_control: &str,
    origin: Option<&str>,
) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Content-Type", content_type)?;
    headers.set("Cache-Control", cache_control)?;
    with_cors(
        Response::from_bytes(body)?
            .with_status(status)
            .with_headers(headers),
        origin,
    )
}

#[cfg(target_arch = "wasm32")]
fn json_response<T: Serialize>(payload: &T, status: u16, origin: Option<&str>) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Content-Type", "application/json; charset=utf-8")?;
    headers.set("Cache-Control", "no-store")?;
    with_cors(
        Response::ok(format!("{}\n", serde_json::to_string(payload)?))?
            .with_status(status)
            .with_headers(headers),
        origin,
    )
}

#[cfg(target_arch = "wasm32")]
fn empty_response(status: u16, origin: Option<&str>) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Cache-Control", "no-store")?;
    with_cors(
        Response::empty()?.with_status(status).with_headers(headers),
        origin,
    )
}

#[cfg(target_arch = "wasm32")]
fn with_cors(mut response: Response, origin: Option<&str>) -> Result<Response> {
    let headers = response.headers_mut();
    headers.set("Access-Control-Allow-Origin", origin.unwrap_or("*"))?;
    headers.set("Vary", "Origin")?;
    headers.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")?;
    headers.set(
        "Access-Control-Allow-Headers",
        "authorization,content-type,range",
    )?;
    headers.set(
        "Access-Control-Expose-Headers",
        "content-range,accept-ranges,content-length",
    )?;
    Ok(response)
}

#[cfg(target_arch = "wasm32")]
fn env_string(env: &worker::Env, name: &str) -> Option<String> {
    env.var(name).map(|value| value.to_string()).ok()
}

#[cfg(target_arch = "wasm32")]
fn env_integer(env: &worker::Env, name: &str, fallback: i64) -> i64 {
    env_string(env, name)
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

#[cfg(target_arch = "wasm32")]
fn js_i64(value: i64) -> std::result::Result<JsValue, HttpError> {
    if value.unsigned_abs() > MAX_SAFE_INTEGER as u64 {
        Err(HttpError::new(
            400,
            "integer exceeds JavaScript safe integer range",
        ))
    } else {
        Ok(JsValue::from_f64(value as f64))
    }
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> std::result::Result<i64, HttpError> {
    let now = js_sys::Date::now();
    if now > MAX_SAFE_INTEGER as f64 {
        Err(HttpError::new(
            500,
            "current timestamp exceeds safe integer range",
        ))
    } else {
        Ok(now as i64)
    }
}

#[cfg(target_arch = "wasm32")]
fn internal_error(error: impl std::fmt::Display) -> HttpError {
    HttpError::new(500, error.to_string())
}

#[cfg(target_arch = "wasm32")]
fn internal_http_error(error: impl std::fmt::Display) -> HttpError {
    HttpError::new(500, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v2_frame(payload_len: u16, frame_count: u16, pts: Option<u64>) -> Vec<u8> {
        let flags = if pts.is_some() { 0x04 } else { 0x00 };
        let word = (0x2b << 26) | (2 << 24) | (flags << 16) | (2 << 12) | (6 << 8) | 4;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(word as u32).to_be_bytes());
        let size_word = ((payload_len as u32) << 16) | frame_count as u32;
        bytes.extend_from_slice(&size_word.to_be_bytes());
        if let Some(pts) = pts {
            bytes.extend_from_slice(&pts.to_be_bytes());
        }
        bytes.extend(std::iter::repeat(0xaa).take(payload_len as usize));
        bytes
    }

    #[test]
    fn scans_frame_stream_offsets_and_fallback_pts() {
        let mut bytes = v2_frame(4, 960, Some(1920));
        let second_offset = bytes.len();
        bytes.extend(v2_frame(3, 960, None));

        let frames = scan_soundkit_frames(&bytes, 100, 0).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].byte_offset, 100);
        assert_eq!(frames[0].start_frame, 1920);
        assert_eq!(frames[0].frame_count, 960);
        assert_eq!(frames[1].byte_offset, 100 + second_offset as i64);
        assert_eq!(frames[1].start_frame, 2880);
        assert_eq!(frames[1].frame_count, 960);
    }

    #[test]
    fn rejects_truncated_frame_stream() {
        let mut bytes = v2_frame(4, 960, None);
        bytes.pop();
        let error = scan_soundkit_frames(&bytes, 0, 0).unwrap_err();
        assert_eq!(error.status, 400);
    }

    #[test]
    fn encodes_sidecar_header_and_entries() {
        let bytes = encode_sidecar_index(48_000, 96_000, &[(0, 0), (100, 960)]);
        assert_eq!(&bytes[0..8], &INDEX_MAGIC);
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[10..12].try_into().unwrap()), 16);
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            48_000
        );
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 2);
        assert_eq!(
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            96_000
        );
        assert_eq!(u64::from_le_bytes(bytes[48..56].try_into().unwrap()), 100);
        assert_eq!(u64::from_le_bytes(bytes[56..64].try_into().unwrap()), 960);
    }
}
