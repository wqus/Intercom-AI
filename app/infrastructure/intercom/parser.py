from __future__ import annotations
import hashlib, hmac, re
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


def _conversation_context(item: dict, current_text: str, current_part: dict | None = None) -> list[dict[str, str]]:
    """Build a safe chronological history for the routing model.

    Only messages that are known to be from the customer or assistant are exposed.
    The current source message and any message newer than that source are excluded
    to prevent delayed webhooks from leaking future conversation state into routing.
    """
    parts = item.get("conversation_parts", {}).get("conversation_parts", []) or []
    context: list[dict[str, str]] = []

    def _stamp(part: dict) -> int | None:
        value = part.get("created_at")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    current_id = None
    if isinstance(current_part, dict):
        current_id = current_part.get("id") or current_part.get("part_id")
    current_stamp = _stamp(current_part or {})
    current_norm = _strip(current_text)

    rows: list[tuple[int, int, dict]] = []
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        body = _strip(part.get("body", ""))
        if not body:
            continue
        part_id = part.get("id") or part.get("part_id")
        stamp = _stamp(part)
        if current_id and part_id and part_id == current_id:
            continue
        if current_part is not None and part is current_part:
            continue
        # If IDs are absent, exclude an exact source-text/timestamp duplicate.
        if current_stamp is not None and stamp is not None and stamp > current_stamp:
            continue
        if current_stamp is not None and stamp == current_stamp and body == current_norm:
            continue
        author_type = (part.get("author", {}) or {}).get("type")
        if author_type not in {"user", "admin", "bot", "teammate", "assistant"}:
            continue
        role = "user" if author_type == "user" else "assistant"
        rows.append((stamp if stamp is not None else -1, idx, {"role": role, "text": body}))

    rows.sort(key=lambda x: (x[0], x[1]))
    context.extend(row[2] for row in rows)
    return context[-12:]

def _conversation(payload: dict) -> tuple[str,str,bool,str,str,str,list[dict[str,str]]] | None:
    item = payload.get("data", {}).get("item", {}) or {}
    cid = str(item.get("id") or "")
    if not cid:
        return None

    source = item.get("source", {}) or {}
    parts = item.get("conversation_parts", {}).get("conversation_parts", []) or []
    topic = payload.get("topic", "")

    # For `conversation.user.replied`, `source` is the message that triggered
    # this webhook. Prefer it over the full conversation history. A delayed or
    # retried webhook can arrive after later user messages are already present in
    # the conversation; selecting the newest historical message would then run
    # the classifier against the wrong text and can bypass an active Top-3 state.
    source_author_type = (source.get("author", {}) or {}).get("type")
    if (
        topic == "conversation.user.replied"
        and (source.get("body") or source.get("attachments"))
        and source_author_type in {"user", None}
    ):
        # The webhook topic itself says this is a customer reply. When the
        # provider omits author.type, the triggering source is still safer than
        # guessing from an older user message in conversation_parts.
        part = source
    else:
        user_parts = [
            p for p in parts
            if (p.get("author", {}) or {}).get("type") == "user" and p.get("body")
        ]
        if user_parts:
            # Compatibility fallback for payloads where the current source is
            # absent or incomplete. Preserve payload order in this fallback.
            part = user_parts[-1]
        elif topic == "conversation.user.replied" and (source.get("body") or source.get("attachments")) and not source_author_type:
            # Compatibility fallback only when author.type is genuinely absent.
            # An explicit admin/bot author must never be treated as a customer reply.
            part = source
        else:
            return None

    text = _strip(part.get("body", ""))
    attachments = bool(part.get("attachments"))
    author = part.get("author", {}) or {}
    last_message_at = part.get("created_at") or item.get("updated_at") or ""
    return cid, text, attachments, author.get("name", ""), author.get("email", ""), str(last_message_at), _conversation_context(item, text, part)

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


def event_order_at(payload: dict) -> int | None:
    """Extract provider-side action time for per-conversation ordering."""
    topic = str(payload.get("topic") or "")
    item = ((payload.get("data") or {}).get("item") or {})
    candidates = []
    if topic.startswith("conversation."):
        candidates.append(((item.get("source") or {}).get("created_at")))
    if topic.startswith("ticket."):
        candidates.append(((item.get("ticket_part") or {}).get("created_at")))
    candidates.append(payload.get("created_at"))
    for value in candidates:
        try:
            if value is not None and str(value).strip() != "":
                return int(value)
        except (TypeError, ValueError):
            continue
    return None

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
    # A message may contain only an attachment. It is still a real user turn
    # and must reach the attachment/escalation policy instead of being silently
    # discarded at the parser boundary.
    if not data or (not data[1] and not data[2]):
        return None
    entity_id,text,attachments,name,email,updated_at,*rest=data
    context = rest[0] if rest and isinstance(rest[0], list) else []
    fallback_key = hashlib.sha1(
        f"{entity_id}:{updated_at}:{text}".encode("utf-8")
    ).hexdigest()
    return InboundMessage(
        event_id=event_id,
        event_key=f"{topic}:{entity_id}:{event_id or fallback_key}",
        topic=topic,
        entity_id=entity_id,
        channel=channel,
        text=text,
        user_name=name,
        user_email=email,
        updated_at=updated_at,
        has_attachments=attachments,
        raw={**payload, "_conversation_context": context},
    )

