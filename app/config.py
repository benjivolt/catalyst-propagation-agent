from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"


def load_dotenv(path: Union[str, Path] = PROJECT_ROOT / ".env") -> None:
    """Small .env loader so local runs do not need python-dotenv."""
    p = Path(path)
    if not p.exists():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()



def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    static_dir: Path = APP_DIR / "static"
    templates_dir: Path = APP_DIR / "templates"
    database_path: str = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "catalyst_propagation.db"))

    
    
    
    openai_api_key: str = _env_first("OPENAI_API_KEY", "OPENAPI_API_KEY", default="")
    # Primary model defaults to the current OpenAI flagship for best reasoning
    # accuracy. The fallback chain is tried only if the primary call fails, so a
    # model deprecation never silently breaks analysis.
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.5")
    openai_healthcheck_model: str = os.getenv("OPENAI_HEALTHCHECK_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.5"))
    # Ordered fallbacks of real, currently-available models:
    # flagship -> prior frontier -> cheap mini -> long-term stable.
    openai_model_fallbacks: str = os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-5.2,gpt-5.4-mini,gpt-4.1-mini")
    openai_strict_analysis: bool = _env_bool("OPENAI_STRICT_ANALYSIS", False)

    
    scraperapi_api_key: str = os.getenv("SCRAPERAPI_API_KEY", "")
    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT", "CatalystPropagationAgent/1.0 contact@example.com"
    )

    monitor_interval_minutes: int = int(os.getenv("MONITOR_INTERVAL_MINUTES", "60"))
    max_sources_per_company: int = int(os.getenv("MAX_SOURCES_PER_COMPANY", "8"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    source_cache_ttl_seconds: int = int(os.getenv("SOURCE_CACHE_TTL_SECONDS", "1800"))

    
    
    
    adaptive_scheduler_autostart: bool = _env_bool("ADAPTIVE_SCHEDULER_AUTOSTART", True)
    scheduler_tick_seconds: int = int(os.getenv("SCHEDULER_TICK_SECONDS", "60"))
    adaptive_min_check_minutes: int = int(os.getenv("ADAPTIVE_MIN_CHECK_MINUTES", "15"))
    adaptive_max_check_hours: int = int(os.getenv("ADAPTIVE_MAX_CHECK_HOURS", "24"))
    adaptive_max_jobs_per_tick: int = int(os.getenv("ADAPTIVE_MAX_JOBS_PER_TICK", "8"))
    scheduler_daily_budget_usd: float = float(os.getenv("SCHEDULER_DAILY_BUDGET_USD", "2.50"))
    scheduler_candidate_threshold: float = float(os.getenv("SCHEDULER_CANDIDATE_THRESHOLD", "0.62"))
    scheduler_llm_threshold: float = float(os.getenv("SCHEDULER_LLM_THRESHOLD", "0.72"))
    scheduler_exploration_factor: float = float(os.getenv("SCHEDULER_EXPLORATION_FACTOR", "0.22"))

    
    
    
    
    source_verification_fetch_on_analyze: bool = _env_bool("SOURCE_VERIFICATION_FETCH_ON_ANALYZE", False)
    source_fetch_retries: int = int(os.getenv("SOURCE_FETCH_RETRIES", "3"))
    source_retry_base_sleep_seconds: float = float(os.getenv("SOURCE_RETRY_BASE_SLEEP_SECONDS", "0.25"))
    source_verification_timeout_seconds: float = float(os.getenv("SOURCE_VERIFICATION_TIMEOUT_SECONDS", "5"))

    
    alerts_enabled: bool = _env_bool("ALERTS_ENABLED", False)
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    alert_email_to: str = os.getenv("ALERT_EMAIL_TO", "")
    alert_email_from: str = os.getenv("ALERT_EMAIL_FROM", "")
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_starttls: bool = _env_bool("SMTP_STARTTLS", True)

    
    
    
    openai_input_cost_per_1m: float = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0"))
    openai_output_cost_per_1m: float = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0"))

    
    external_scheduler_worker: bool = _env_bool("EXTERNAL_SCHEDULER_WORKER", False)


settings = Settings()
