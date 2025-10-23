from flask import Flask
from flask_cors import CORS
from chat_app import chat_bp
from report_app import report_bp
from dashboard import dashboard_bp

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(chat_bp, url_prefix="/chat")
app.register_blueprint(report_bp, url_prefix="/reports")
app.register_blueprint(dashboard_bp, url_prefix="/dashboard")

import logging
import sys

# ---------------- Logging Setup ----------------
# Configure root logger to stdout so Gunicorn/systemd capture it
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Tie Flask logger to Gunicorn logger if available
gunicorn_logger = logging.getLogger("gunicorn.error")
if gunicorn_logger.handlers:
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

app.logger.info("✅ Flask logging initialized and connected to Gunicorn")

@app.route("/ping")
def ping():
    app.logger.info("Ping route hit!")
    print("Ping route hit — plain print() works too")
    return "pong"


if __name__ == "__main__":
    print("✅ Llama3 Banking Platform (Dashboard + Chat + Detailed Reports)")
    app.run(port=8004, debug=True)
