# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PII redaction for agent/dialog text: paths (Win/Unix), emails, secrets, IDs.
# Configurable and extensible for multi-platform and custom patterns.

import re
from typing import List, Optional, Tuple

from data_juicer.ops.base_op import OPERATORS, Mapper

OP_NAME = "pii_redaction_mapper"

PLACEHOLDER_PATH = "[PATH_REDACTED]"
PLACEHOLDER_EMAIL = "[EMAIL_REDACTED]"
PLACEHOLDER_SECRET = "[REDACTED]"
PLACEHOLDER_ID = "[ID_REDACTED]"
PLACEHOLDER_PHONE = "[PHONE_REDACTED]"
PLACEHOLDER_ID_CARD = "[ID_CARD_REDACTED]"


def _compile_patterns() -> dict:
    """Build regex patterns for multi-platform and common PII."""
    return {
        # Unix: /Users/..., /home/..., /tmp/..., /opt/...
        "path_unix": re.compile(
            r"(^|[\s\"'(\[=])(/(?:Users?|home|tmp|etc|var|opt|Applications)" r"[^\s\"')\]]*(?:/[^\s\"')\]]*)*)"
        ),
        # Windows: C:\, D:\, \\server\share
        "path_win": re.compile(r"(^|[\s\"'(\[=])([A-Za-z]:\\[^\s\"')\]]*(?:\\[^\s\"')\]]*)*)"),
        "path_win_unc": re.compile(r"(^|[\s\"'(\[=])(\\\\[^\s\"')\]]+)"),
        "email": re.compile(r"[\w.+-]+@[\w.-]+\.\w+"),
        # Secret keys: key:= value or "key": "value"
        "secret_kv": re.compile(
            r"(\b(?:api[_-]?key|apikey|secret|password|passwd|token|auth|authorization"
            r"|credential|license[_-]?key)\s*[:=]\s*[\"']?)([^\s\"',}\]]+)",
            re.IGNORECASE,
        ),
        # Session/user/request IDs
        "id_kv": re.compile(
            r"\b(session[_-]?id|user[_-]?id|request[_-]?id|trace[_-]?id)\s*[:=]\s*[\"']?[\w.-]+",
            re.IGNORECASE,
        ),
        # Chinese mobile (simple)
        "phone_cn": re.compile(r"\b1[3-9]\d{9}\b"),
        # Generic phone (international, simple)
        "phone_intl": re.compile(r"\+\d{1,4}[-.\s]?\d{6,14}\b"),
        # Chinese ID card (18 digits)
        "id_card_cn": re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"),
    }


@OPERATORS.register_module(OP_NAME)
class PiiRedactionMapper(Mapper):
    """Redact PII in text: paths (Unix/Windows), emails, secrets, IDs, phones.

    All options are configurable; supports custom extra patterns for
    multi-platform and domain-specific PII.
    """

    def __init__(
        self,
        mask_paths: bool = True,
        mask_emails: bool = True,
        mask_secrets: bool = True,
        mask_ids: bool = True,
        mask_phones: bool = False,
        mask_id_cards: bool = False,
        path_replacement: str = PLACEHOLDER_PATH,
        email_replacement: str = PLACEHOLDER_EMAIL,
        secret_replacement: str = PLACEHOLDER_SECRET,
        id_replacement: str = PLACEHOLDER_ID,
        phone_replacement: str = PLACEHOLDER_PHONE,
        id_card_replacement: str = PLACEHOLDER_ID_CARD,
        extra_patterns: Optional[List[Tuple[str, str]]] = None,
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(text_key=text_key, **kwargs)
        self.mask_paths = mask_paths
        self.mask_emails = mask_emails
        self.mask_secrets = mask_secrets
        self.mask_ids = mask_ids
        self.mask_phones = mask_phones
        self.mask_id_cards = mask_id_cards
        self.path_replacement = path_replacement
        self.email_replacement = email_replacement
        self.secret_replacement = secret_replacement
        self.id_replacement = id_replacement
        self.phone_replacement = phone_replacement
        self.id_card_replacement = id_card_replacement
        self.extra_patterns = extra_patterns or []

        patterns = _compile_patterns()
        self._path_unix = patterns["path_unix"]
        self._path_win = patterns["path_win"]
        self._path_win_unc = patterns["path_win_unc"]
        self._email_re = patterns["email"]
        self._secret_kv = patterns["secret_kv"]
        self._id_kv = patterns["id_kv"]
        self._phone_cn = patterns["phone_cn"]
        self._phone_intl = patterns["phone_intl"]
        self._id_card_cn = patterns["id_card_cn"]

        self._extra_compiled = []
        for pat_str, repl in self.extra_patterns:
            try:
                self._extra_compiled.append((re.compile(pat_str), repl))
            except re.error:
                pass

    def _redact_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        if self.mask_paths:
            text = self._path_unix.sub(r"\1" + self.path_replacement, text)
            text = self._path_win.sub(r"\1" + self.path_replacement, text)
            text = self._path_win_unc.sub(r"\1" + self.path_replacement, text)
        if self.mask_emails:
            text = self._email_re.sub(self.email_replacement, text)
        if self.mask_secrets:
            text = self._secret_kv.sub(r"\1" + self.secret_replacement, text)
        if self.mask_ids:
            text = self._id_kv.sub(r"\1" + self.id_replacement, text)
        if self.mask_phones:
            text = self._phone_cn.sub(self.phone_replacement, text)
            text = self._phone_intl.sub(self.phone_replacement, text)
        if self.mask_id_cards:
            text = self._id_card_cn.sub(self.id_card_replacement, text)
        for pat, repl in self._extra_compiled:
            text = pat.sub(repl, text)
        return text

    def process_single(self, sample):
        if self.text_key in sample and sample[self.text_key]:
            sample[self.text_key] = self._redact_text(sample[self.text_key])
        return sample
