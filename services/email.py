import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USER)
APP_URL = os.getenv('APP_URL', 'http://localhost:5000')

def send_email(to_email, subject, html_body):
    if not SMTP_USER or not SMTP_PASSWORD:
        raise ValueError("SMTP not configured. Set SMTP_USER and SMTP_PASSWORD in .env")

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = f"LearnForge <{FROM_EMAIL}>"
    msg['To'] = to_email
    msg.attach(MIMEText(html_body, 'html'))

    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_email, msg.as_string())
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_email, msg.as_string())
    return True

def send_verification_email(to_email, name, token):
    verify_url = f"{APP_URL}/api/auth/verify-email/{token}"

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

    return send_email(to_email, "Verify your LearnForge account", html)

def send_password_reset_email(to_email, name, token):
    reset_url = f"{APP_URL}/resetpassword?token={token}"

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

    return send_email(to_email, "Reset your LearnForge password", html)