# backend/streaming_logic.py
import asyncio
import logging
from threading import Thread
import cv2
from aiohttp import web
from .media_processor import StandardProcessor # <-- Import the swappable processor

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 8080
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
latest_frame = None

def video_processing_worker():
    global latest_frame
    # To switch to your ULANC processor later, you would change this one line:
    # processor = ULANCProcessor()
    processor = StandardProcessor()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("FATAL: Could not open video source.")
        return
    logging.info("Successfully opened camera. Video processing is running...")
    
    while True:
        ret, img = cap.read()
        if not ret: break
        
        output_img = processor.process_frame(img)
        
        ret, buffer = cv2.imencode('.jpg', output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ret:
            latest_frame = buffer.tobytes()
    cap.release()

async def video_feed(request):
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=--frame'
    await response.prepare(request)
    while True:
        try:
            if latest_frame:
                await response.write(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            await asyncio.sleep(1/30)
        except (ConnectionResetError, ConnectionAbortedError):
            break
    return response

async def index(request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ULANC Processed Stream</title>
        <style>
            body { font-family: sans-serif; background-color: #1a1a1a; color: #e0e0e0; text-align: center; margin: 0; padding: 20px;}
            img { background-color: #000; border: 2px solid #444; width: 80%; max-width: 960px; }
        </style>
    </head>
    <body>
        <h1>ULANC Processed Stream</h1>
        <p>This is the live, AI-processed video from the Python server.</p>
        <img src="/video_feed" alt="Video Stream">
    </body>
    </html>
    """
    return web.Response(content_type='text/html', text=html_content)

def run_server():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/video_feed", video_feed)
    
    processing_thread = Thread(target=video_processing_worker, daemon=True)
    processing_thread.start()
    
    logging.info(f"Starting server at http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)