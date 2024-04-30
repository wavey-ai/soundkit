use h3::server::RequestStream;
use http::{Method, Request, Response, StatusCode};
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnectionBuilder;
use pki_types::{CertificateDer, PrivateKeyDer};
use regex::Regex;
use rustls::{Certificate, PrivateKey};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::{fs::File, io, io::BufReader};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::time::{sleep, Duration, Instant};
use tokio_rustls::TlsAcceptor;
use tracing::{error, info};
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3;

pub struct HyperServer {
    ssl_path: String,
    ssl_port: u16,
}

impl HyperServer {
    pub fn new(ssl_path: String, ssl_port: u16) -> Self {
        Self { ssl_path, ssl_port }
    }

    pub async fn start(
        &self,
    ) -> Result<tokio::sync::watch::Sender<()>, Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = watch::channel(());

        let addr = SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), self.ssl_port);

        let crt_path = format!("{}/{}", self.ssl_path, "cert.pem");
        let key_path = format!("{}/{}", self.ssl_path, "privkey.pem");

        let crt_path = Path::new(&crt_path);
        let key_path = Path::new(&key_path);

        let certs = load_certs(crt_path).unwrap();
        let key = load_keys(key_path).unwrap();

        let mut server_config = tokio_rustls::rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)?;
        server_config.alpn_protocols =
            vec![b"h2".to_vec(), b"http/1.1".to_vec(), b"http/1.0".to_vec()];
        let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));

        println!("Starting to serve on https://{}", addr);

        let srv_h2 = {
            let mut shutdown_signal = rx.clone();
            async move {
                let incoming = TcpListener::bind(&addr).await.unwrap();
                let service = service_fn(move |req| handle_request_h2(req));

                loop {
                    tokio::select! {
                        _ = shutdown_signal.changed() => {
                            break;
                        }
                        result = incoming.accept() => {
                            let (tcp_stream, _remote_addr) = result.unwrap();
                            let tls_acceptor = tls_acceptor.clone();
                            let service = service.clone();

                            tokio::spawn(async move {
                                let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                                    Ok(tls_stream) => tls_stream,
                                    Err(err) => {
                                        eprintln!("failed to perform tls handshake: {err:#}");
                                        return;
                                    }
                                };
                                if let Err(err) = ConnectionBuilder::new(TokioExecutor::new())
                                    .serve_connection(TokioIo::new(tls_stream), service)
                                    .await
                                {
                                    eprintln!("failed to serve connection: {err:#}");
                                }
                            });
                        }
                    }
                }
            }
        };

        let certs =
            Certificate(std::fs::read(format!("{}/{}", self.ssl_path, "cert.der")).unwrap());
        let key =
            PrivateKey(std::fs::read(format!("{}/{}", self.ssl_path, "privkey.der")).unwrap());

        let mut tls_config = rustls::ServerConfig::builder()
            .with_safe_default_cipher_suites()
            .with_safe_default_kx_groups()
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![certs], key)
            .unwrap();

        tls_config.max_early_data_size = u32::MAX;
        tls_config.alpn_protocols = vec![b"h3".to_vec()];

        let server_config = quinn::ServerConfig::with_crypto(Arc::new(tls_config));
        let addr = SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), self.ssl_port + 1);
        let endpoint = quinn::Endpoint::server(server_config, addr).unwrap();

        let srv_h3 = {
            let mut shutdown_signal = rx.clone();

            async move {
                loop {
                    tokio::select! {
                        _ = shutdown_signal.changed() => {
                                break;
                        }
                        res = endpoint.accept()  => {
                            if let Some(new_conn) = res {
                                info!("New connection being attempted");

                                tokio::spawn(async move {
                                    match new_conn.await {
                                        Ok(conn) => {
                                            let mut h3_conn = h3::server::Connection::new(h3_quinn::Connection::new(conn)).await.unwrap();

                                            loop {
                                                match h3_conn.accept().await {
                                                    Ok(Some((req, stream))) => {
                                                        info!("new request: {:#?}", req);

                                                        tokio::spawn(async move {
                                                        if let Err(e) = handle_request_h3(req, stream).await {
                                                            error!("handling request failed: {}", e);
                                                            }
                                                        });
                                                    }

                                                    Ok(None) => {
                                                        break;
                                                    },
                                                    Err(err) => {
                                                    error!("error on accept {}", err);
                                                    break;
                                                    }
                                                }

                                            }
                                        }
                                        Err(err) => {
                                            error!("accepting connection failed: {:?}", err);
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }
        };

        tokio::spawn(srv_h2);
        tokio::spawn(srv_h3);

        Ok(tx)
    }
}

async fn handle_request_h2(
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Box<dyn std::error::Error + Send + Sync>> {
    let (status, data, content_type) =
        request_handler(req.method(), req.uri().path(), req.uri().query()).await?;
    if let (Some(data), Some(content_type)) = (data, content_type) {
        let mut response = Response::new(Full::from(data.0));
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert("alt-srv", "h3=\":4433\"; ma=2592000".parse().unwrap());

        response
            .headers_mut()
            .insert("content-type", content_type.parse().unwrap());
        response
            .headers_mut()
            .insert("etag", format!("{}", data.1).parse().unwrap());

        if content_type == "application/vnd.apple.mpegurl" {
            response
                .headers_mut()
                .insert("content-encoding", "gzip".parse().unwrap());
            response
                .headers_mut()
                .insert("vary", "accept-encoding".parse().unwrap());
        }
        Ok(response)
    } else {
        let mut response = Response::new(Full::default());
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert("alt-srv", "h3=\":4433\"; ma=2592000".parse().unwrap());
        Ok(response)
    }
}

async fn handle_request_h3(
    req: Request<()>,
    mut stream: RequestStream<h3_quinn::BidiStream<Bytes>, Bytes>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (status, data, content_type) =
        request_handler(req.method(), req.uri().path(), req.uri().query()).await?;

    if let (Some(data), Some(content_type)) = (data, content_type) {
        let mut r = http::Response::builder()
            .status(status)
            .header("content-type", content_type.clone())
            .header("etag", data.1);
        if content_type == "application/vnd.apple.mpegurl" {
            r = r
                .header("content-encoding", "gzip")
                .header("vary", "accept-encoding");
        }
        let resp = r.body(()).unwrap();

        match stream.send_response(resp).await {
            Ok(_) => {}
            Err(err) => {
                error!("unable to send response to connection peer: {:?}", err);
            }
        }

        stream.send_data(data.0).await?;
    } else {
        let resp = http::Response::builder()
            .status(status)
            .header("content-type", "text/plain")
            .body(())
            .unwrap();

        match stream.send_response(resp).await {
            Ok(_) => {}
            Err(err) => {
                error!("unable to send response to connection peer: {:?}", err);
            }
        }
    }

    Ok(stream.finish().await?)
}

async fn request_handler(
    method: &Method,
    path: &str,
    query: Option<&str>,
) -> Result<
    (StatusCode, Option<(Bytes, u64)>, Option<String>),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let res = match (method, path) {
        (&Method::OPTIONS, _) => (StatusCode::OK, None, None),

        (&Method::GET, path) => {
            let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
            let keys = match path_segments.as_slice() {
                [segment1, segment2] => vec![segment1.to_string(), segment2.to_string()],
                _ => vec![],
            };

            if keys.is_empty() {
                (StatusCode::NOT_FOUND, None, None)
            } else if keys[1] == "up" || keys[0] == "up" {
                (
                    StatusCode::OK,
                    Some((Bytes::from("OK"), 0)),
                    Some("text/plain".into()),
                )
            } else {
                (StatusCode::NOT_FOUND, None, None)
            }
        }
        _ => (StatusCode::NOT_FOUND, None, None),
    };

    Ok(res)
}

fn add_cors_headers(res: &mut http::Response<http_body_util::Full<Bytes>>) {
    res.headers_mut()
        .insert("access-control-allow-origin", "*".parse().unwrap());
    res.headers_mut().insert(
        "access-control-allow-methods",
        "GET, POST, PUT, DELETE, OPTIONS".parse().unwrap(),
    );
    res.headers_mut().insert(
        "access-control-allow-headers",
        "Content-Type".parse().unwrap(),
    );
}

fn load_certs(path: &Path) -> io::Result<Vec<CertificateDer<'static>>> {
    certs(&mut BufReader::new(File::open(path)?)).collect()
}

fn load_keys(path: &Path) -> io::Result<PrivateKeyDer<'static>> {
    pkcs8_private_keys(&mut BufReader::new(File::open(path)?))
        .next()
        .unwrap()
        .map(Into::into)
}
