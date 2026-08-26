from __future__ import annotations
import hashlib, hmac, re
from datetime import datetime
from app.core.config import settings
from app.domain.enums import Channel
from app.domain.models import InboundMessage

CONVERSATION_TOPICS={"conversation.user.created","conversation.user.replied"}
TICKET_TOPICS={"ticket.created","ticket.contact.replied"}

def verify_signature(body: bytes, signature: str) -> bool:
    if not settings.INTERCOM_CLIENT_SECRET:
        return False
    expected="sha1="+hmac.new(settings.INTERCOM_CLIENT_SECRET.encode(),body,hashlib.sha1).hexdigest()
    return hmac.compare_digest(expected, signature or "")

def _strip(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()

def _conversation(payload: dict) -> tuple[str,str,bool,str,str,str] | None:
    item = payload.get("data", {}).get("item", {}) or {}
    cid = str(item.get("id") or "")
    if not cid:
        return None

    source = item.get("source", {}) or {}
    parts = item.get("conversation_parts", {}).get("conversation_parts", []) or []
    topic = payload.get("topic", "")

    # Intercom exposes the conversation as history, where the first customer
    # message is still present after later replies arrive. Classification must
    # operate on the LATEST customer message. Prefer the freshest timestamp
    # across `source` and `conversation_parts`; when timestamps are missing,
    # retain payload order and take the last user part.
    user_parts = [
        p for p in parts
        if (p.get("author", {}) or {}).get("type") == "user" and p.get("body")
    ]
    if (source.get("author", {}) or {}).get("type") == "user" and source.get("body"):
        user_parts.append(source)

    def _timestamp(value: object) -> float:
        if value in (None, ""):
            return -1.0
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError, OverflowError):
                return -1.0

    if user_parts:
        part = max(
            enumerate(user_parts),
            key=lambda pair: (_timestamp(pair[1].get("created_at")), pair[0]),
        )[1]
    elif topic == "conversation.user.replied" and source.get("body") and not (source.get("author", {}) or {}).get("type"):
        # Compatibility fallback only when author.type is genuinely absent. An
        # explicit admin/bot author must never be treated as a customer reply.
        part = source
    else:
        return None

    text = _strip(part.get("body", ""))
    attachments = bool(part.get("attachments"))
    author = part.get("author", {}) or {}
    last_message_at = part.get("created_at") or item.get("updated_at") or ""
    return cid, text, attachments, author.get("name", ""), author.get("email", ""), str(last_message_at)

def _ticket(payload: dict) -> tuple[str,str,bool,str,str,str] | None:
    item=payload.get("data",{}).get("item",{})
    topic=payload.get("topic","")
    if topic=="ticket.contact.replied":
        ticket=item.get("ticket",{}) or {}
        tid=str(ticket.get("id") or item.get("id") or "")
        part=item.get("ticket_part",{}) or {}
        text=_strip(part.get("body",""))
        author=part.get("author",{}) or {}
        attachments=bool(part.get("attachments"))
        last_message_at = part.get("created_at") or item.get("updated_at") or ""
        return tid,text,attachments,author.get("name",""),author.get("email",""),str(last_message_at)
    tid=str(item.get("id") or "")
    attrs=item.get("ticket_attributes",{}) or {}
    text=_strip(attrs.get("_default_description_",""))
    return (tid,text,False,"","",str(item.get("updated_at",""))) if tid else None

def parse(payload: dict) -> InboundMessage | None:
    topic=payload.get("topic","")
    event_id=str(payload.get("id") or "")
    if topic in CONVERSATION_TOPICS:
        data=_conversation(payload)
        channel=Channel.CONVERSATION
    elif topic in TICKET_TOPICS:
        data=_ticket(payload)
        channel=Channel.TICKET
    else:
        return None
    if not data or not data[1]:
        return None
    entity_id,text,attachments,name,email,updated_at=data
    return InboundMessage(
        event_id=event_id,
        event_key=f"{topic}:{entity_id}:{event_id or hashlib.sha1(text.encode()).hexdigest()}",
        topic=topic,
        entity_id=entity_id,
        channel=channel,
        text=text,
        user_name=name,
        user_email=email,
        updated_at=updated_at,
        has_attachments=attachments,
        raw=payload,
    )

