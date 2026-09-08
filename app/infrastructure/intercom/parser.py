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

def _conversation(payload: dict) -> tuple[str,str,bool,str,str,str,list[dict[str,str]],str|None] | None:
    item = payload.get("data", {}).get("item", {}) or {}
    cid = str(item.get("id") or "")
    if not cid:
        return None

    source = item.get("source", {}) or {}
    parts = item.get("conversation_parts", {}).get("conversation_parts", []) or []
    topic = payload.get("topic", "")
    notification_created = None
    try:
        notification_created = int(payload.get("created_at"))
    except (TypeError, ValueError):
        pass

    def _part_time(part: dict) -> int | None:
        try:
            return int(part.get("created_at")) if part.get("created_at") is not None else None
        except (TypeError, ValueError):
            return None

    source_author_type = (source.get("author", {}) or {}).get("type")
    source_body = _strip(source.get("body", ""))
    source_attachments = bool(source.get("attachments"))

    part = None
    provider_part_id = None

    # Intercom's conversation.user.replied payloads are not completely uniform:
    # sometimes `source` is the conversation source object (e.g. admin_initiated)
    # and the actual triggering customer message is present in conversation_parts.
    # When source itself is an explicit user message, it is authoritative.
    if topic in {"conversation.user.created", "conversation.user.replied"} and source_author_type in {"user", None} and (source_body or source_attachments):
        part = source
        provider_part_id = source.get("id") or source.get("part_id")

    if part is None and topic in {"conversation.user.created", "conversation.user.replied"}:
        # Prefer the newest user part that is not newer than the webhook notification.
        # This avoids replaying a later message already present in Intercom history.
        candidates = []
        for idx, candidate in enumerate(parts):
            if not isinstance(candidate, dict):
                continue
            author_type = (candidate.get("author", {}) or {}).get("type")
            if author_type != "user":
                continue
            body = _strip(candidate.get("body", ""))
            attachments = bool(candidate.get("attachments"))
            if not body and not attachments:
                continue
            stamp = _part_time(candidate)
            if notification_created is not None and stamp is not None and stamp > notification_created:
                continue
            # When source is an explicit admin/bot object, do not accidentally select
            # a stale user part unless its timestamp is consistent with this event.
            candidates.append((stamp if stamp is not None else -1, idx, candidate))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            part = candidates[-1][2]
            provider_part_id = part.get("id") or part.get("part_id")

    # Compatibility for malformed payloads where source.author.type is absent.
    # Trust source only if it is clearly message-like; otherwise keep the bounded
    # history fallback above.
    if part is None:
        return None

    text = _strip(part.get("body", ""))
    attachments = bool(part.get("attachments"))
    author = part.get("author", {}) or {}
    last_message_at = part.get("created_at") or item.get("updated_at") or ""
    context = _conversation_context(item, text, part)

    # If source itself has no part id, but the matching historical part does, use
    # that provider part id as a cross-topic idempotency anchor. This is critical
    # when Intercom emits both conversation.user.replied and ticket.contact.replied
    # for the same customer comment.
    if not provider_part_id:
        target_stamp = _part_time(part)
        for candidate in parts:
            if not isinstance(candidate, dict):
                continue
            if _strip(candidate.get("body", "")) != text:
                continue
            if target_stamp is not None and _part_time(candidate) != target_stamp:
                continue
            candidate_author = (candidate.get("author", {}) or {}).get("type")
            if candidate_author == "user" and candidate.get("id"):
                provider_part_id = candidate.get("id")
                break

    return cid, text, attachments, author.get("name", ""), author.get("email", ""), str(last_message_at), context, (str(provider_part_id) if provider_part_id else None)

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
        return tid,text,attachments,author.get("name",""),author.get("email",""),str(last_message_at),[],str(part.get("id") or part.get("part_id") or "") or None
    tid=str(item.get("id") or "")
    attrs=item.get("ticket_attributes",{}) or {}
    text=_strip(attrs.get("_default_description_",""))
    return (tid,text,False,"","",str(item.get("updated_at","")),[],None) if tid else None


def event_order_at(payload: dict) -> int | None:
    """Extract the timestamp of the actual triggering customer action.

    Intercom conversation webhooks may expose an unrelated conversation `source`
    object (for example `admin_initiated`) even when the topic is
    `conversation.user.replied`. The ordering watermark must therefore prefer the
    triggering user part, not blindly trust `source.created_at`.
    """
    topic = str(payload.get("topic") or "")
    item = ((payload.get("data") or {}).get("item") or {})

    def _as_int(value):
        try:
            if value is not None and str(value).strip() != "":
                return int(value)
        except (TypeError, ValueError):
            return None
        return None

    candidates: list[int] = []

    if topic == "ticket.contact.replied":
        value = _as_int(((item.get("ticket_part") or {}).get("created_at")))
        if value is not None:
            candidates.append(value)

    if topic in {"conversation.user.created", "conversation.user.replied"}:
        source = item.get("source") or {}
        source_author_type = (source.get("author") or {}).get("type")
        source_has_message = bool(_strip(source.get("body", "")) or source.get("attachments"))
        # Trust source.created_at when source is message-like and is not an
        # explicitly administrative source. This also covers malformed payloads
        # where Intercom omits author.type but source carries the user message.
        if source_author_type not in {"admin", "bot", "teammate", "assistant"} and source_has_message:
            value = _as_int(source.get("created_at"))
            if value is not None:
                candidates.append(value)

        # Real user message parts are the authoritative fallback. Prefer the
        # newest user part at or before the notification creation timestamp.
        notification_created = _as_int(payload.get("created_at"))
        parts = (item.get("conversation_parts") or {}).get("conversation_parts", []) or []
        part_times: list[int] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if (part.get("author") or {}).get("type") != "user":
                continue
            if not (_strip(part.get("body", "")) or part.get("attachments")):
                continue
            value = _as_int(part.get("created_at"))
            if value is None:
                continue
            if notification_created is not None and value > notification_created:
                continue
            part_times.append(value)
        if part_times:
            candidates.append(max(part_times))

    if not candidates:
        for value in (payload.get("created_at"), item.get("updated_at")):
            parsed = _as_int(value)
            if parsed is not None:
                candidates.append(parsed)

    return max(candidates) if candidates else None

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
    provider_part_id = rest[1] if len(rest) > 1 else None
    fallback_key = hashlib.sha1(
        f"{entity_id}:{updated_at}:{text}".encode("utf-8")
    ).hexdigest()
    dedup_anchor = provider_part_id or fallback_key
    event_key = (
        f"message:{entity_id}:{dedup_anchor}"
        if provider_part_id
        else f"{topic}:{entity_id}:{event_id or fallback_key}"
    )
    return InboundMessage(
        event_id=event_id,
        event_key=event_key,
        topic=topic,
        entity_id=entity_id,
        channel=channel,
        text=text,
        user_name=name,
        user_email=email,
        updated_at=updated_at,
        has_attachments=attachments,
        raw={**payload, "_conversation_context": context, "_provider_part_id": provider_part_id},
    )

