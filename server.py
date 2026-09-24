"""
Utility module: server.py.
"""

from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    port = 8080
    print(f"Starting server on http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    server = HTTPServer(('0.0.0.0', port), CORSRequestHandler)
    server.serve_forever()
