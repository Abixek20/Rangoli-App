"""
WSGI entry point for production deployment.
Run with: gunicorn wsgi:application -w 2 -b 0.0.0.0:$PORT
"""
from app import app as application

if __name__ == "__main__":
    application.run()
