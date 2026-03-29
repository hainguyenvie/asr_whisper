"""
Audio Upload and Streaming Endpoints
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import httpx
import os
import tempfile
import subprocess
import json

router = APIRouter()

# Whisper.cpp server URL
WHISPER_SERVER_URL = os.getenv("WHISPER_SERVER_URL", "https://netmind.viettel.vn/zipformer_stt_hainh67")

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    meeting_id: str = Form(...),
    employee_code: str = Form(...)
):
    """
    Upload complete audio file for transcription
    """
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Send to Whisper STT server (OpenAI-compatible endpoint)
        async with httpx.AsyncClient(timeout=3600.0) as client:
            with open(temp_path, "rb") as f:
                response = await client.post(
                    f"{WHISPER_SERVER_URL}/v1/audio/transcriptions",
                    files={"file": f},
                    data={
                        "model": "whisper-1",
                        "response_format": "verbose_json",
                        "diarization": "true",  # Full upload - high quality with speakers
                        "temperature": 0.0
                    }
                )

        # Cleanup
        os.unlink(temp_path)

        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "meeting_id": meeting_id,
                "transcript": result.get("text", ""),
                "segments": result.get("segments", [])
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Whisper server error: {response.text}"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    meeting_id: str = Form(...),
    employee_code: str = Form(...)
):
    """
    Upload audio chunk for real-time processing
    This is called when WebSocket is not available
    """
    try:
        # Save chunk temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Send to Whisper for transcription (fast mode for chunks)
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(temp_path, "rb") as f:
                response = await client.post(
                    f"{WHISPER_SERVER_URL}/v1/audio/transcriptions",
                    files={"file": f},
                    data={
                        "model": "whisper-1",
                        "response_format": "verbose_json",
                        "diarization": "false",  # Chunk mode - fast, no speaker labels
                        "temperature": 0.0
                    }
                )

        # Cleanup
        os.unlink(temp_path)

        if response.status_code == 200:
            result = response.json()
            
            # The STT server now returns 'segments'. We should use them.
            # But the frontend expects a single transcript update for this chunk.
            # We'll take the dominant info.
            
            segments = result.get("segments", [])
            transcript_text = ""
            speaker = "Unknown"
            
            valid_segments = [s for s in segments if s.get('text', '').strip()]
            
            if valid_segments:
                # Join text (clean, without speaker labels if they are separate field)
                # Note: stt_server's seg['text'] is RAW text. stt_server's result['text'] is formatted with [SPEAKER].
                # We want raw text here and let frontend format it, or use the formatted one?
                # The user wants "Avatar + Text". So raw text + speaker ID is best.
                
                transcript_text = " ".join([s.get('text', '') for s in valid_segments])
                
                # Pick the speaker of the longest segment or just the first one
                # For a chunk (usually small), likely one speaker.
                speaker = valid_segments[0].get('speaker', 'Unknown')
                if speaker == "UNKNOWN": speaker = "Unknown" 
                
            else:
                # Fallback
                transcript_text = result.get("text", "")
                # Clean brackets if fallback used
                import re
                transcript_text = re.sub(r'\[SPEAKER_.*?\]:\s*', '', transcript_text)

            if not transcript_text:
                transcript_text = "" # Avoid null

            return {
                "text": transcript_text,
                "speaker": speaker,
                "timestamp": result.get("timestamp", 0),
                "meeting_id": meeting_id
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to transcribe chunk"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    employee_code: str = Form(...)
):
    """
    Simple transcription endpoint (no meeting association)
    """
    try:
        # Save file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Call Whisper STT (simple transcription)
        async with httpx.AsyncClient(timeout=3600.0) as client:
            with open(temp_path, "rb") as f:
                response = await client.post(
                    f"{WHISPER_SERVER_URL}/v1/audio/transcriptions",
                    files={"file": f},
                    data={
                        "model": "whisper-1",
                        "response_format": "verbose_json",
                        "diarization": "false",  # Simple mode - just transcription
                        "temperature": 0.0
                    }
                )

        # Cleanup
        os.unlink(temp_path)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Transcription failed"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
