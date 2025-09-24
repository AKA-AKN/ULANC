# client.py
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import logging
import argparse
import json
from aiohttp import ClientSession

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- U-LANC Video Processing Logic ---
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Define blur kernels for different quality levels
QUALITY_LEVELS = {
    "high": (21, 21),   # Low blur for good networks
    "medium": (61, 61), # Medium blur
    "low": (101, 101)   # Heavy blur for poor networks
}

class UlanCVideoTrack(MediaStreamTrack):
    """
    A video track that applies the U-LANC adaptive compression effect.
    """
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.current_quality = "high" # Start with high quality

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # --- Adaptive Compression Logic ---
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = segmentation.process(rgb_img)
        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.1

        kernel = QUALITY_LEVELS[self.current_quality]
        blurred_background = cv2.GaussianBlur(img, kernel, 0)

        output_img = np.where(condition, img, blurred_background)
        # --- End of Logic ---

        # Rebuild the video frame
        new_frame = VideoFrame.from_ndarray(output_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

async def monitor_network(pc, video_track):
    """
    Monitors the network stats and updates the video quality.
    """
    while True:
        await asyncio.sleep(5) # Check every 5 seconds
        try:
            stats = await pc.getStats()
            for report in stats.values():
                if report["type"] == "candidate-pair" and report.get("state") == "succeeded":
                    available_bitrate = report.get("availableOutgoingBitrate")
                    if available_bitrate:
                        if available_bitrate > 1_000_000: # > 1 Mbps
                            if video_track.current_quality != "high":
                                logging.info("Network is GOOD. Setting quality to HIGH.")
                                video_track.current_quality = "high"
                        elif available_bitrate < 400_000: # < 400 Kbps
                            if video_track.current_quality != "low":
                                logging.info("Network is POOR. Setting quality to LOW.")
                                video_track.current_quality = "low"
                        else:
                            if video_track.current_quality != "medium":
                                logging.info("Network is OK. Setting quality to MEDIUM.")
                                video_track.current_quality = "medium"
        except Exception as e:
            logging.error(f"Error getting stats: {e}")


async def run(pc, role, signaling_server):
    """
    Main function to run the WebRTC client.
    """
    session = ClientSession()

    @pc.on("track")
    def on_track(track):
        logging.info(f"Track {track.kind} received")
        # Display the received video
        async def display_track():
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                cv2.imshow("Received Video", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        asyncio.ensure_future(display_track())

    if role == "send":
        # Create a video source from the webcam
        from aiortc.contrib.media import MediaPlayer
        player = MediaPlayer("video=Integrated Camera", format="dshow", options={"video_size": "640x480"})
        
        # Create the adaptive video track and add it to the peer connection
        video_track = UlanCVideoTrack(player.video)
        pc.addTrack(video_track)

        # Start the network monitor
        asyncio.ensure_future(monitor_network(pc, video_track))

        # Create offer
        await pc.setLocalDescription(await pc.createOffer())
        offer = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        
        # Send offer to signaling server
        async with session.post(f"{signaling_server}/offer", json=offer) as resp:
            logging.info(f"Offer sent, server responded with {await resp.text()}")

        # Wait for answer
        while True:
            logging.info("Waiting for answer...")
            await asyncio.sleep(2)
            async with session.get(f"{signaling_server}/answer") as resp:
                if resp.status == 200:
                    answer_json = await resp.json()
                    if answer_json:
                        answer = RTCSessionDescription(sdp=answer_json["sdp"], type=answer_json["type"])
                        await pc.setRemoteDescription(answer)
                        logging.info("Answer received and connection established.")
                        break
    else: # Role is "receive"
        # Wait for offer
        while True:
            logging.info("Waiting for offer...")
            await asyncio.sleep(2)
            async with session.get(f"{signaling_server}/offer") as resp:
                 if resp.status == 200:
                    offer_json = await resp.json()
                    if offer_json:
                        offer = RTCSessionDescription(sdp=offer_json["sdp"], type=offer_json["type"])
                        await pc.setRemoteDescription(offer)
                        
                        # Create answer
                        await pc.setLocalDescription(await pc.createAnswer())
                        answer = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

                        # Send answer to signaling server
                        async with session.post(f"{signaling_server}/answer", json=answer) as resp_post:
                            logging.info(f"Answer sent, server responded with {await resp_post.text()}")
                        break
    
    # Keep the application running
    try:
        await asyncio.Event().wait()
    finally:
        await pc.close()
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-LANC WebRTC Client")
    parser.add_argument("role", choices=["send", "receive"])
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Signaling server URL")
    args = parser.parse_args()

    pc = RTCPeerConnection()
    try:
        asyncio.run(run(pc, args.role, args.server))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
