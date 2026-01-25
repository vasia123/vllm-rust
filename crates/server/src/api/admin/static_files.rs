//! Static file serving for embedded frontend assets.

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

/// Embedded frontend assets from the build output.
/// Path is relative to the crate root (crates/server/).
#[derive(RustEmbed)]
#[folder = "../../frontend/dist"]
#[prefix = ""]
pub struct FrontendAssets;

/// Serve an embedded static file.
pub async fn serve_static(path: &str) -> Response {
    let path = if path.is_empty() || path == "/" {
        "index.html"
    } else {
        path.trim_start_matches('/')
    };

    match FrontendAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            let body = Body::from(content.data.to_vec());
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(body)
                .unwrap()
        }
        None => {
            // Try index.html for SPA routing (client-side routes)
            if !path.contains('.') {
                if let Some(content) = FrontendAssets::get("index.html") {
                    let body = Body::from(content.data.to_vec());
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "text/html")
                        .body(body)
                        .unwrap();
                }
            }
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Not Found"))
                .unwrap()
        }
    }
}

/// Handler for static file requests.
pub async fn static_handler(
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    serve_static(&path).await
}

/// Handler for root index.
pub async fn index_handler() -> impl IntoResponse {
    serve_static("").await
}
