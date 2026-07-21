use std::{
    sync::{Arc, OnceLock},
    time::Instant,
};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{stream, StreamExt};
use reqwest::Client;
use tracing::{debug, error};

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{
        is_retryable_status, AttachedBody, ConnectionMode, RetryExecutor, RuntimeType, Worker,
        WorkerLoadGuard, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::GenerationRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::{RerankRequest, RerankResponse, RerankResult},
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        error::{self, extract_error_code_from_response},
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
};

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    dp_aware: bool,
    enable_igw: bool,
    retry_config: RetryConfig,
}

const MOL_INTERNAL_HOP_KEY: &str = "mol_internal_hop";
const MOL_ROUTE_HOP: &str = "route";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MolRouteSource {
    None,
    Header,
    LegacyMetadata,
}

/// Request-local cache for runtime-specific route payloads. The normal path
/// borrows the original JSON for vLLM and allocates only when SGLang is chosen.
struct WorkerPayloadCache<'a> {
    serialized: &'a serde_json::Value,
    source: MolRouteSource,
    sglang: OnceLock<Result<serde_json::Value, String>>,
    legacy_passthrough: OnceLock<Result<serde_json::Value, String>>,
}

impl<'a> WorkerPayloadCache<'a> {
    fn new(
        headers: Option<&HeaderMap>,
        serialized: &'a serde_json::Value,
        route: &str,
    ) -> Result<Self, String> {
        let header_hop = header_utils::extract_mol_internal_hop(headers);
        let metadata_hop_value = serialized
            .get("metadata")
            .and_then(serde_json::Value::as_object)
            .and_then(|metadata| metadata.get(MOL_INTERNAL_HOP_KEY));
        let metadata_hop = metadata_hop_value
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| "internal hop metadata value must be a string".to_string())
            })
            .transpose()?;

        if header_hop.is_some() && metadata_hop.is_some() && header_hop != metadata_hop {
            return Err("conflicting internal hop header and metadata values".to_string());
        }

        let internal_hop = header_hop.or(metadata_hop);
        if let Some(internal_hop) = internal_hop {
            if internal_hop != MOL_ROUTE_HOP {
                return Err("unsupported internal hop value".to_string());
            }
            if route != "/v1/chat/completions" {
                return Err("mol route marker is valid only for chat completions".to_string());
            }
        }

        let source = if metadata_hop.is_some() {
            MolRouteSource::LegacyMetadata
        } else if header_hop.is_some() {
            MolRouteSource::Header
        } else {
            MolRouteSource::None
        };

        Ok(Self {
            serialized,
            source,
            sglang: OnceLock::new(),
            legacy_passthrough: OnceLock::new(),
        })
    }

    fn payload_for(&self, runtime_type: &RuntimeType) -> Result<&serde_json::Value, String> {
        match (self.source, runtime_type) {
            (MolRouteSource::None, _) | (MolRouteSource::Header, RuntimeType::Vllm) => {
                Ok(self.serialized)
            }
            (MolRouteSource::Header, RuntimeType::External) => Ok(self.serialized),
            (MolRouteSource::Header, RuntimeType::Sglang)
            | (MolRouteSource::LegacyMetadata, RuntimeType::Sglang) => {
                Self::cached(&self.sglang, || {
                    let mut payload = self.serialized.clone();
                    let root = payload
                        .as_object_mut()
                        .ok_or_else(|| "chat payload must be a JSON object".to_string())?;
                    if self.source == MolRouteSource::LegacyMetadata {
                        Router::remove_legacy_internal_metadata(root)?;
                    }
                    Router::adapt_sglang_route_payload(root)?;
                    Ok(payload)
                })
            }
            (MolRouteSource::LegacyMetadata, _) => Self::cached(&self.legacy_passthrough, || {
                let mut payload = self.serialized.clone();
                let root = payload
                    .as_object_mut()
                    .ok_or_else(|| "chat payload must be a JSON object".to_string())?;
                Router::remove_legacy_internal_metadata(root)?;
                Ok(payload)
            }),
        }
    }

    fn cached(
        cell: &OnceLock<Result<serde_json::Value, String>>,
        initialize: impl FnOnce() -> Result<serde_json::Value, String>,
    ) -> Result<&serde_json::Value, String> {
        match cell.get_or_init(initialize) {
            Ok(payload) => Ok(payload),
            Err(message) => Err(message.clone()),
        }
    }
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("dp_aware", &self.dp_aware)
            .field("enable_igw", &self.enable_igw)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    /// Create a new router with injected policy and client
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            dp_aware: ctx.router_config.dp_aware,
            enable_igw: ctx.router_config.enable_igw,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    /// Select worker for a specific model considering circuit breaker state
    async fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        // Get workers for the specified model O(1), filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            effective_model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let idx = policy
            .select_worker(
                &available,
                &SelectWorkerInfo {
                    request_text: text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await?;

        // Record worker selection metric (Layer 3)
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            policy.name(),
        );

        Some(available[idx].clone())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
    ) -> Response {
        let serialized = match serde_json::to_value(typed_req) {
            Ok(value) => value,
            Err(err) => {
                return error::bad_request(
                    "serialization_failed",
                    format!("Convert into serde_json::Value failed: {err}"),
                )
            }
        };
        let text = typed_req.extract_text_for_routing();
        self.route_serialized_request(
            headers,
            &serialized,
            route,
            model_id,
            typed_req.is_stream(),
            &text,
        )
        .await
    }

    async fn route_serialized_request(
        &self,
        headers: Option<&HeaderMap>,
        serialized: &serde_json::Value,
        route: &'static str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let worker_payloads = match WorkerPayloadCache::new(headers, serialized, route) {
            Ok(payloads) => payloads,
            Err(message) => {
                return error::bad_request("invalid_internal_route_payload", message);
            }
        };
        let start = Instant::now();
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let res = self
                    .route_serialized_request_once(
                        headers,
                        &worker_payloads,
                        route,
                        model_id,
                        is_stream,
                        text,
                    )
                    .await;

                // Need to be outside `route_typed_request_once` because that function has multiple return paths
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    res.status().as_u16(),
                    extract_error_code_from_response(&res),
                );

                res
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                // Layer 3 worker metrics
                Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // on_exhausted hook
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            },
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_serialized_request_once(
        &self,
        headers: Option<&HeaderMap>,
        worker_payloads: &WorkerPayloadCache<'_>,
        route: &'static str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let worker = match self
            .select_worker_for_model(model_id, Some(text), headers)
            .await
        {
            Some(w) => w,
            None => {
                return error::service_unavailable(
                    "no_available_workers",
                    "No available workers (all circuits open or unhealthy)",
                );
            }
        };

        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        let load_guard = ["cache_aware", "manual"]
            .contains(&policy.name())
            .then(|| WorkerLoadGuard::new(worker.clone(), headers));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let worker_payload = match worker_payloads.payload_for(&worker.metadata().runtime_type) {
            Ok(payload) => payload,
            Err(message) => {
                return error::bad_request("invalid_internal_route_payload", message);
            }
        };

        let response = self
            .send_json_request(
                headers,
                worker_payload,
                route,
                &worker,
                is_stream,
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        // For streaming responses, the wrapped body (`BreakerTrackedStream`)
        // records the circuit-breaker outcome once the stream actually
        // terminates (success on clean end, failure on mid-stream error).
        // Recording it eagerly here based on the initial status code would
        // mask "200-then-broken" workers — every request would tick a
        // success before the stream had a chance to error out.
        if !is_stream {
            worker.record_outcome(status.is_success());
        }

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    fn remove_legacy_internal_metadata(
        root: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        let remove_metadata = {
            let metadata = root
                .get_mut("metadata")
                .and_then(serde_json::Value::as_object_mut)
                .ok_or_else(|| "metadata must be a JSON object".to_string())?;
            metadata.remove(MOL_INTERNAL_HOP_KEY);
            metadata.is_empty()
        };
        if remove_metadata {
            root.remove("metadata");
        }
        Ok(())
    }

    fn adapt_sglang_route_payload(
        root: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        let messages = root
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or_else(|| "route messages must be an array".to_string())?;

        let assistant = messages
            .last()
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| "route payload must end with an assistant prefill".to_string())?;
        if assistant.get("role").and_then(serde_json::Value::as_str) != Some("assistant") {
            return Err("route payload must end with an assistant prefill".to_string());
        }
        let prefill = assistant
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "assistant prefill content must be a string".to_string())?
            .to_string();
        messages.pop();

        let user = messages
            .last_mut()
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| "route payload must contain a user message".to_string())?;
        if user.get("role").and_then(serde_json::Value::as_str) != Some("user") {
            return Err("assistant prefill must follow a user message".to_string());
        }
        let user_content = user
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "route user content must be a string".to_string())?;
        let mut combined = user_content.trim_end().to_string();
        combined.push('\n');
        combined.push_str(&prefill);
        user.insert("content".to_string(), serde_json::Value::String(combined));

        root.remove("continue_final_message");
        root.remove("add_generation_prompt");
        root.remove("stop");
        root.insert(
            "separate_reasoning".to_string(),
            serde_json::Value::Bool(false),
        );

        let template_kwargs = root
            .entry("chat_template_kwargs".to_string())
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
            .ok_or_else(|| "chat_template_kwargs must be a JSON object".to_string())?;
        template_kwargs.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );

        Ok(())
    }

    // Helper: return base worker URL (strips DP suffix when enabled)
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.dp_aware {
            if let Ok((prefix, _)) = Self::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| header_utils::should_forward_request_header(name.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let worker_url = worker.url();
                let base = self.worker_base_url(worker_url);
                let url = format!("{}/{}", base, endpoint);
                let client = self.client.clone();
                let method = method.clone();

                let headers = filtered_headers.clone();

                let api_key = worker.api_key().clone();

                async move {
                    let mut request_builder = match method {
                        Method::GET => client.get(url),
                        Method::POST => client.post(url),
                        _ => {
                            return Err(error::method_not_allowed(
                                "unsupported_method",
                                "Unsupported method for simple routing",
                            ))
                        }
                    };

                    if let Some(key) = api_key {
                        let mut auth_header = String::with_capacity(7 + key.len());
                        auth_header.push_str("Bearer ");
                        auth_header.push_str(&key);
                        request_builder = request_builder.header("Authorization", auth_header);
                    }

                    for (name, value) in headers {
                        request_builder = request_builder.header(name.clone(), value.clone());
                    }

                    request_builder.send().await.map_err(convert_reqwest_error)
                }
            })
            .collect();

        // Now execute the collected futures concurrently
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut last_response: Option<Response> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;

                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(e);
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // TODO (rui): Better accommodate to the Worker abstraction
    fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
        let parts: Vec<&str> = worker_url.split('@').collect();
        if parts.len() != 2 {
            return Err(format!("invalid worker_url format: {}", worker_url));
        }

        // Parse the second part (dp_rank) into an integer
        match parts[1].parse::<usize>() {
            Ok(dp_rank) => Ok((parts[0], dp_rank)),
            Err(_) => Err(format!(
                "failed to parse dp_rank from worker_url: {}",
                worker_url
            )),
        }
    }

    // Send typed request directly without conversion
    async fn send_json_request(
        &self,
        headers: Option<&HeaderMap>,
        serialized: &serde_json::Value,
        route: &'static str,
        worker: &Arc<dyn Worker>,
        is_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        let worker_url = worker.url();
        let api_key = worker.api_key().clone();

        // Static key string to avoid per-request allocations
        const DP_RANK_KEY: &str = "data_parallel_rank";

        let mut request_builder = if self.dp_aware {
            let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return error::internal_error(
                        "dp_rank_extraction_failed",
                        format!("Failed to extract dp_rank: {}", e),
                    );
                }
            };

            let mut json_val = serialized.clone();

            if let Some(map) = json_val.as_object_mut() {
                // Use static key string to avoid allocation
                map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
                // Only serialize if debug logging is enabled to avoid CPU overhead
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!(
                        "Modified request body: {}",
                        serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
                    );
                }
            } else {
                return error::bad_request(
                    "dp_rank_insertion_failed",
                    "Failed to insert the data_parallel_rank field into the request body",
                );
            }

            self.client
                .post(format!("{}{}", worker_url_prefix, route))
                .json(&json_val)
        } else {
            self.client
                .post(format!("{}{}", worker_url, route))
                .json(serialized)
        };

        if let Some(key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                // For streaming requests the caller skips the eager
                // `record_outcome` on the assumption that a
                // `BreakerTrackedStream` will tick the breaker on drop —
                // but no tracked stream is installed when send() fails
                // before any response stream exists. Record the failure
                // here so a worker flapping at the TCP layer doesn't
                // stay permanently selectable. Non-streaming requests
                // are already covered by the caller's
                // `worker.record_outcome(status.is_success())`, so
                // gating on `is_stream` avoids double-counting.
                if is_stream {
                    worker.record_outcome(false);
                }
                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        } else {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            // Pass the reqwest byte stream straight through as the response body.
            // Dropping the response body drops this stream, which closes the
            // upstream HTTP connection and lets the engine abort generation —
            // no spawned task or channel needed. `BreakerTrackedStream`
            // updates the worker's circuit breaker exactly once on drop:
            // success on clean end, failure on stream error, neither on
            // client disconnect. For non-2xx responses we pre-mark the
            // wrapper as Errored — otherwise the small error body would
            // stream cleanly to `None` and Drop would record a spurious
            // success (and the streaming branch also skips the eager
            // `record_outcome` above).
            let mut tracked = BreakerTrackedStream::new(
                res.bytes_stream(),
                worker.clone(),
                worker_url.to_string(),
            );
            if !status.is_success() {
                tracked.mark_errored();
            }
            let body = Body::from_stream(tracked);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            if let Some(guard) = load_guard {
                response = AttachedBody::wrap_response(response, guard);
            }
            response
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_chat_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        raw_body: &serde_json::Value,
        model_id: Option<&str>,
    ) -> Response {
        let text = body.extract_text_for_routing();
        self.route_serialized_request(
            headers,
            raw_body,
            "/v1/chat/completions",
            model_id,
            body.is_stream(),
            &text,
        )
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/classify", model_id)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return error::internal_error(
                        "rerank_response_build_failed",
                        "Failed to build rerank response",
                    );
                }
            }
        } else {
            response
        }
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    fn route_payload() -> serde_json::Value {
        serde_json::json!({
            "model": "L0",
            "messages": [
                {"role": "user", "content": "classify this request"},
                {"role": "assistant", "content": "model_id="}
            ],
            "max_tokens": 24,
            "max_completion_tokens": 24,
            "temperature": 0.0,
            "stream": false,
            "continue_final_message": true,
            "add_generation_prompt": false,
            "stop": null,
            "structured_outputs": {"choice": ["L0", "L1", "L2", "L3"]}
        })
    }

    fn route_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-mol-internal-hop", "route".parse().unwrap());
        headers
    }

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .build();
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            dp_aware: false,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            enable_igw: false,
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_healthy(false);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_unmarked_chat_payload_is_not_cloned_or_adapted() {
        let payload = serde_json::json!({
            "model": "L0",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let cache = WorkerPayloadCache::new(None, &payload, "/v1/chat/completions").unwrap();
        let worker_payload = cache.payload_for(&RuntimeType::Sglang).unwrap();
        assert!(std::ptr::eq(worker_payload, &payload));
    }

    #[test]
    fn test_vllm_header_route_payload_is_borrowed_without_clone() {
        let payload = route_payload();
        let headers = route_headers();
        let cache =
            WorkerPayloadCache::new(Some(&headers), &payload, "/v1/chat/completions").unwrap();
        let worker_payload = cache.payload_for(&RuntimeType::Vllm).unwrap();

        assert!(std::ptr::eq(worker_payload, &payload));
        assert_eq!(worker_payload["continue_final_message"], true);
        assert_eq!(worker_payload["messages"][1]["content"], "model_id=");
    }

    #[test]
    fn test_sglang_route_payload_is_converted_once_and_cached() {
        let payload = route_payload();
        let headers = route_headers();
        let cache =
            WorkerPayloadCache::new(Some(&headers), &payload, "/v1/chat/completions").unwrap();
        let adapted = cache.payload_for(&RuntimeType::Sglang).unwrap();
        let cached = cache.payload_for(&RuntimeType::Sglang).unwrap();

        assert!(std::ptr::eq(adapted, cached));
        assert!(adapted.get("metadata").is_none());
        assert_eq!(adapted["messages"].as_array().unwrap().len(), 1);
        assert_eq!(
            adapted["messages"][0],
            serde_json::json!({
                "role": "user",
                "content": "classify this request\nmodel_id="
            })
        );
        assert!(adapted.get("continue_final_message").is_none());
        assert!(adapted.get("add_generation_prompt").is_none());
        assert!(adapted.get("stop").is_none());
        assert_eq!(adapted["separate_reasoning"], false);
        assert_eq!(adapted["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(
            adapted["structured_outputs"],
            serde_json::json!({"choice": ["L0", "L1", "L2", "L3"]})
        );
    }

    #[test]
    fn test_other_metadata_survives_internal_marker_removal() {
        let mut payload = route_payload();
        payload["metadata"] = serde_json::json!({
            "mol_internal_hop": "route",
            "trace": "keep-me"
        });
        let cache = WorkerPayloadCache::new(None, &payload, "/v1/chat/completions").unwrap();
        let adapted = cache.payload_for(&RuntimeType::Vllm).unwrap();
        assert_eq!(adapted["metadata"], serde_json::json!({"trace": "keep-me"}));
    }

    #[test]
    fn test_legacy_metadata_route_is_still_supported_for_sglang() {
        let mut payload = route_payload();
        payload["metadata"] = serde_json::json!({"mol_internal_hop": "route"});
        let cache = WorkerPayloadCache::new(None, &payload, "/v1/chat/completions").unwrap();
        let adapted = cache.payload_for(&RuntimeType::Sglang).unwrap();

        assert!(adapted.get("metadata").is_none());
        assert_eq!(adapted["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_invalid_or_conflicting_internal_hop_is_rejected() {
        let payload = route_payload();
        let mut invalid_headers = HeaderMap::new();
        invalid_headers.insert("x-mol-internal-hop", "answer".parse().unwrap());
        assert!(
            WorkerPayloadCache::new(Some(&invalid_headers), &payload, "/v1/chat/completions")
                .is_err()
        );

        let mut legacy_payload = payload;
        legacy_payload["metadata"] = serde_json::json!({"mol_internal_hop": "route"});
        assert!(WorkerPayloadCache::new(
            Some(&invalid_headers),
            &legacy_payload,
            "/v1/chat/completions"
        )
        .is_err());

        legacy_payload["metadata"] = serde_json::json!({"mol_internal_hop": 1});
        assert!(WorkerPayloadCache::new(None, &legacy_payload, "/v1/chat/completions").is_err());
    }
}
