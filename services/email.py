"""
Email service using SMTP.
Simple approach - direct SMTP connection for each email.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()


class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_user)
        self.app_url = os.getenv('APP_URL', 'http://localhost:5000')

    def _send_email(self, to_email, subject, html_body):
        """Send email via SMTP."""
        if not self.smtp_user or not self.smtp_password:
            raise ValueError("SMTP not configured. Set SMTP_USER and SMTP_PASSWORD in .env")

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"LearnForge <{self.from_email}>"
        msg['To'] = to_email

        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)

        try:
            # Use SSL for port 465, STARTTLS for port 587
            if self.smtp_port == 465:
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as server:
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.from_email, to_email, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.from_email, to_email, msg.as_string())
            return True
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send to {to_email}: {e}")
            raise e

    def send_verification_email(self, to_email, name, token):
        """Send email verification link."""
        verify_url = f"{self.app_url}/api/auth/verify-email/{token}"

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">LearnForge</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <h2 style="color: #333;">Welcome, {name}!</h2>
                <p style="color: #666; font-size: 16px;">
                    Thanks for signing up! Please verify your email address to complete your registration.
                </p>
                <p style="text-align: center; margin: 30px 0;">
                    <a href="{verify_url}"
                       style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                              color: white; padding: 14px 28px;
                              text-decoration: none; border-radius: 6px; font-weight: bold;
                              display: inline-block;">
                        Verify Email Address
                    </a>
                </p>
                <p style="color: #999; font-size: 12px;">
                    If the button doesn't work, copy this link:<br>
                    <a href="{verify_url}" style="color: #667eea;">{verify_url}</a>
                </p>
            </div>
        </body>
        </html>
        """

        return self._send_email(to_email, "Verify your LearnForge account", html)

    def send_password_reset_email(self, to_email, name, token):
        """Send password reset link."""
        reset_url = f"{self.app_url}/resetpassword?token={token}"

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">LearnForge</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <h2 style="color: #333;">Password Reset Request</h2>
                <p style="color: #666; font-size: 16px;">
                    Hi {name}, you requested to reset your password. Click the button below to set a new password:
                </p>
                <p style="text-align: center; margin: 30px 0;">
                    <a href="{reset_url}"
                       style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                              color: white; padding: 14px 28px;
                              text-decoration: none; border-radius: 6px; font-weight: bold;
                              display: inline-block;">
                        Reset Password
                    </a>
                </p>
                <p style="color: #999; font-size: 12px;">
                    This link expires in 1 hour.<br>
                    If you didn't request this, please ignore this email.
                </p>
                <p style="color: #999; font-size: 12px;">
                    If the button doesn't work, copy this link:<br>
                    <a href="{reset_url}" style="color: #667eea;">{reset_url}</a>
                </p>
            </div>
        </body>
        </html>
        """

        return self._send_email(to_email, "Reset your LearnForge password", html)
