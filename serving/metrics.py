from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

# Prometheus Metrics Definitions
HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total model predictions made by outcome",
    ["prediction_class"],
)

PREDICTION_PROBABILITY_HISTOGRAM = Histogram(
    "model_prediction_probability",
    "Histogram of predicted default probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ACTIVE_MODEL_VERSION = Gauge(
    "model_active_version_info",
    "Active model serving information",
    ["model_name", "version", "source"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to measure HTTP request count and latency."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        endpoint = request.url.path
        
        # Don't track the /metrics endpoint itself to avoid recursion
        if endpoint == "/metrics":
            return await call_next(request)

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            HTTP_REQUEST_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
            raise e

        latency = time.time() - start_time
        HTTP_REQUEST_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(latency)

        return response


def get_metrics_response() -> Response:
    """Returns the latest Prometheus metrics in text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
