"""POST /chat — run one conversational turn through the LangChain agent."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .voice_agent import ChatTurnResult, VoiceAgent

router = APIRouter()
log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    transcript: str = Field(..., description="The user's transcribed speech.")
    thread_id: str | None = Field(
        None, description="Continue an existing conversation, or omit to start fresh."
    )


class ConfirmRequest(BaseModel):
    thread_id: str
    approved: bool
    modification_request: str | None = None


class PendingConfirmationPayload(BaseModel):
    tool_name: str
    tool_args: dict
    confirmation_prompt: str


class ChatResponse(BaseModel):
    text: str
    emotion: str
    thread_id: str
    needs_followup: bool
    pending_confirmation: PendingConfirmationPayload | None = None


def _get_voice_agent(request: Request) -> VoiceAgent:
    va = getattr(request.app.state, "voice_agent", None)
    if va is None:
        raise HTTPException(status_code=503, detail="Voice agent not ready")
    return va


def _to_response(result: ChatTurnResult) -> ChatResponse:
    pc = None
    if result.pending_confirmation is not None:
        pc = PendingConfirmationPayload(
            tool_name=result.pending_confirmation.tool_name,
            tool_args=result.pending_confirmation.tool_args,
            confirmation_prompt=result.pending_confirmation.confirmation_prompt,
        )
    return ChatResponse(
        text=result.text,
        emotion=result.emotion,
        thread_id=result.thread_id,
        needs_followup=result.needs_followup,
        pending_confirmation=pc,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    agent = _get_voice_agent(request)
    try:
        result = agent.chat(req.transcript, req.thread_id)
    except Exception as exc:
        log.exception("chat handler failed for transcript=%r", req.transcript)
        # Surface the cause to the client so it shows up in the Reachy log,
        # not just in the server console. Don't leak the full traceback in
        # the message, but do include the exception class + message.
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    return _to_response(result)


@router.post("/chat/confirm", response_model=ChatResponse)
def confirm(req: ConfirmRequest, request: Request) -> ChatResponse:
    agent = _get_voice_agent(request)
    try:
        result = agent.confirm(
            req.thread_id,
            approved=req.approved,
            modification_request=req.modification_request,
        )
    except Exception as exc:
        log.exception("chat/confirm handler failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    return _to_response(result)
