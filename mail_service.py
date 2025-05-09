import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

def send_email(name,email, question):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    support_email = os.getenv("SUPPORT_EMAIL")
    
    print("🔌 Connecting to SMTP server...")

    # Construct email
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = support_email
    msg["Subject"] = f"Support Request from {name}"

    body = f"""
    Name: {name}
    Email: {email}
    Question: {question}
    """
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect and send
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            print("✅ Email sent successfully")
            server.quit()
    except Exception as e:
        print("❌ Failed to send email:", e)


