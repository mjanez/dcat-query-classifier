"""Internationalization (i18n) module for SPARQL Humanize.

This module provides translation functionality using gettext.
"""

import gettext
from pathlib import Path
from typing import Optional

from .config import config


class I18n:
    """Internationalization handler."""

    def __init__(self, locale: Optional[str] = None) -> None:
        """Initialize i18n with specified locale.

        Args:
            locale: Locale code (e.g., 'en', 'es'). If None, uses config default.
        """
        self.locale = locale or config.get("i18n.default_locale", "en")
        self.supported_locales = config.get("i18n.supported_locales", ["en"])
        self.translations_dir = self._get_translations_dir()

        # Validate locale
        if self.locale not in self.supported_locales:
            self.locale = config.get("i18n.default_locale", "en")

        self._translator = self._load_translations()

    def _get_translations_dir(self) -> Path:
        """Get the translations directory path.

        Returns:
            Path to translations directory.
        """
        project_root = Path(__file__).parent.parent
        translations_dir = project_root / config.get("i18n.translations_dir", "locales")
        translations_dir.mkdir(exist_ok=True)
        return translations_dir

    def _load_translations(self) -> gettext.GNUTranslations:
        """Load translations for current locale.

        Returns:
            Translations object.
        """
        try:
            return gettext.translation(
                domain="sparql_humanize",
                localedir=str(self.translations_dir),
                languages=[self.locale],
                fallback=True,
            )
        except Exception:
            # Fallback to NullTranslations if translations not found
            return gettext.NullTranslations()

    def gettext(self, message: str) -> str:
        """Get translated message.

        Args:
            message: Message to translate.

        Returns:
            Translated message or original if translation not found.
        """
        return self._translator.gettext(message)

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Get translated message with plural support.

        Args:
            singular: Singular form of message.
            plural: Plural form of message.
            n: Count to determine singular/plural.

        Returns:
            Translated message in appropriate form.
        """
        return self._translator.ngettext(singular, plural, n)

    def set_locale(self, locale: str) -> None:
        """Change the current locale.

        Args:
            locale: New locale code.
        """
        if locale in self.supported_locales:
            self.locale = locale
            self._translator = self._load_translations()


# Global i18n instance
_i18n: Optional[I18n] = None


def get_i18n(locale: Optional[str] = None) -> I18n:
    """Get or create global i18n instance.

    Args:
        locale: Locale code. If None, uses config default.

    Returns:
        I18n instance.
    """
    global _i18n
    if _i18n is None or (locale and locale != _i18n.locale):
        _i18n = I18n(locale)
    return _i18n


def _(message: str) -> str:
    """Shortcut function for translation.

    Args:
        message: Message to translate.

    Returns:
        Translated message.
    """
    return get_i18n().gettext(message)
