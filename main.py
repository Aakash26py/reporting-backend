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

if __name__ == "__main__":
    print("✅ Llama3 Banking Platform (Dashboard + Chat + Detailed Reports)")
    app.run(port=8004, debug=True)
