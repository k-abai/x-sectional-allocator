import smtplib
from email.message import EmailMessage
from core.logger import SystemLogger
import os

class Notifier:
    def __init__(self, config):
        self.config = config.get('notifications', {})
        self.logger = SystemLogger()
        self.email_enabled = self.config.get('email_enabled', False)
        
    def send_email(self, subject: str, body: str):
        if not self.email_enabled:
            return
            
        to_email = self.config.get('to_email') or os.getenv("EMAIL_TO")
        from_email = self.config.get('from_email') or os.getenv("EMAIL_FROM")
        
        smtp_server = os.getenv("EMAIL_HOST", "localhost")
        try:
            smtp_port = int(os.getenv("EMAIL_PORT", 587))
        except:
            smtp_port = 587
            
        smtp_user = os.getenv("EMAIL_USERNAME")
        smtp_pass = os.getenv("EMAIL_PASSWORD")
        use_tls = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
        
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        try:
            if smtp_server == "localhost":
                self.logger.log_event("EMAIL_MOCK_SENT", {"to": to_email, "subject": subject, "body": body})
                return
                
            with smtplib.SMTP(smtp_server, smtp_port) as s:
                if use_tls:
                    s.starttls()
                if smtp_user and smtp_pass:
                    s.login(smtp_user, smtp_pass)
                s.send_message(msg)
            self.logger.log_event("EMAIL_SENT", {"to": to_email, "subject": subject})
        except Exception as e:
            self.logger.log_error(f"Email failed: {str(e)}")

    def send_kill_alert(self):
        subject = "[ALERT] KILL SWITCH ACTIVATED"
        body = "The system kill switch has been activated. All positions closed and orders cancelled."
        self.send_email(subject, body)

    def send_regime_alert(self, symbol, reason):
        subject = f"[ALERT] Risk Off - {symbol}"
        body = f"Strategy triggered risk-off.\nReason: {reason}"
        self.send_email(subject, body)
