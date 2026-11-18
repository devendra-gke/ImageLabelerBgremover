import os
import sys
import threading

from django.apps import AppConfig


class ProcessorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'processor'

    def ready(self):
        """Load the AI models in the background as soon as the server starts,
        so the first uploaded image does not wait 10-30 s for model loading.
        Disable with WARMUP_MODELS=0."""
        if os.getenv("WARMUP_MODELS", "1") == "0":
            return
        argv = " ".join(sys.argv)
        is_server = "runserver" in argv or "gunicorn" in argv or "waitress" in argv or "uvicorn" in argv
        # runserver's auto-reloader starts two processes; warm only the worker.
        if "runserver" in argv and "--noreload" not in argv and os.environ.get("RUN_MAIN") != "true":
            return
        if not is_server:
            return

        def _warm():
            from .services.image_processor import load_preset_settings
            from .services.model_sessions import (
                resolve_detection_model,
                resolve_model_name,
                warm_up,
            )

            try:
                bg = load_preset_settings("jewellery").get("background_removal", {})
            except Exception:
                bg = {}
            warm_up([resolve_detection_model(bg), resolve_model_name(bg)])
            print("[warm-up] AI models ready")

        threading.Thread(target=_warm, name="model-warmup", daemon=True).start()
