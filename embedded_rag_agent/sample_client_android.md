# Integrating with Android / Web clients (Notes)

Clients can implement a chat UI that supports:
- Text messages from the server (JSON 'text' field)
- Attachments: server returns 'attachments' list, where each item has 'type' and 'path'. The client should fetch the file via provided path, e.g. GET /download/{filename}
- For images: show inline thumbnails and allow tap-to-download/open
- For PDFs: fetch and open with PDF viewer on device

Security:
- Use HTTPS inside your network or a secure internal reverse proxy.
- API tokens: implement simple header-based token checks in app.py for production.

Example Android flow:
1. POST /chat with JSON input -> receive text + attachments
2. If attachments returned, fetch /download/filename with authorization header and display.
