# server.py
import asyncio
import json
import logging
from aiohttp import web
import aiohttp_cors

# Configure logging
logging.basicConfig(level=logging.INFO)

# A simple in-memory storage for the SDP offer and answer
offer = None
answer = None

async def handle_offer(request):
    """ Handles the POST request for the offer. """
    global offer
    if request.method == "POST":
        data = await request.json()
        offer = data
        logging.info("Offer received and stored.")
        return web.Response(text="Offer stored", status=200)
    elif request.method == "GET":
        logging.info("Offer requested.")
        return web.json_response(offer)

async def handle_answer(request):
    """ Handles the POST request for the answer. """
    global answer
    if request.method == "POST":
        data = await request.json()
        answer = data
        logging.info("Answer received and stored.")
        return web.Response(text="Answer stored", status=200)
    elif request.method == "GET":
        logging.info("Answer requested.")
        return web.json_response(answer)

# Create the web application
app = web.Application()

# Configure CORS for all routes
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
})

# Add routes
app.router.add_route("*", "/offer", handle_offer)
app.router.add_route("*", "/answer", handle_answer)

# Add CORS to all routes
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    logging.info("Starting signaling server at http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)