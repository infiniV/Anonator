"""Auto-update system for Anonator launcher and app."""

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict
from packaging import version

from .config import VERSION, UPDATE_CHECK_URL
from .utils import download_file

logger = logging.getLogger(__name__)


class UpdateChecker:
    """Check for and download application updates."""

    def __init__(self, current_version: str = VERSION):
        """Initialize update checker.

        Args:
            current_version: Current version string
        """
        self.current_version = current_version
        self.update_url = UPDATE_CHECK_URL

    def check_for_updates(self) -> Optional[Dict]:
        """Check if updates are available.

        Returns:
            Dictionary with update info or None if no update available
        """
        try:
            logger.info(f"Checking for updates (current version: {self.current_version})")

            # Fetch latest release info from GitHub API
            with urllib.request.urlopen(self.update_url, timeout=10) as response:
                data = json.loads(response.read().decode())

            latest_version = data.get('tag_name', '').lstrip('v')

            if not latest_version:
                logger.warning("Could not determine latest version")
                return None

            # Compare versions
            if version.parse(latest_version) > version.parse(self.current_version):
                logger.info(f"Update available: {latest_version}")

                return {
                    'version': latest_version,
                    'name': data.get('name', ''),
                    'description': data.get('body', ''),
                    'download_url': self._get_download_url(data),
                    'published_at': data.get('published_at', ''),
                }
            else:
                logger.info("No updates available")
                return None

        except urllib.error.URLError as e:
            logger.warning(f"Failed to check for updates: {e}")
            return None
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return None

    def _get_download_url(self, release_data: Dict) -> Optional[str]:
        """Extract download URL for launcher from release assets.

        Args:
            release_data: GitHub release data

        Returns:
            Download URL or None
        """
        assets = release_data.get('assets', [])

        # Look for launcher executable
        for asset in assets:
            name = asset.get('name', '')
            if 'launcher' in name.lower() and name.endswith('.exe'):
                return asset.get('browser_download_url')

        return None

    def download_update(
        self,
        update_info: Dict,
        destination: Path,
        progress_callback=None
    ) -> Optional[Path]:
        """Download update package.

        Args:
            update_info: Update information from check_for_updates
            destination: Where to save downloaded file
            progress_callback: Optional callback(downloaded, total)

        Returns:
            Path to downloaded file or None if failed
        """
        download_url = update_info.get('download_url')

        if not download_url:
            logger.error("No download URL in update info")
            return None

        try:
            logger.info(f"Downloading update from {download_url}")

            downloaded_file = download_file(
                download_url,
                destination,
                progress_callback
            )

            logger.info(f"Update downloaded to {downloaded_file}")
            return downloaded_file

        except Exception as e:
            logger.error(f"Failed to download update: {e}")
            return None

    def apply_update(self, update_file: Path) -> bool:
        """Apply downloaded update.

        This is a placeholder - actual implementation would:
        1. Verify update signature
        2. Backup current installation
        3. Replace launcher executable
        4. Restart launcher

        Args:
            update_file: Path to downloaded update

        Returns:
            True if successful
        """
        # TODO: Implement update application logic
        # For now, just log
        logger.info(f"Would apply update from {update_file}")
        return False


def check_and_notify_updates() -> Optional[Dict]:
    """Convenience function to check for updates.

    Returns:
        Update info if available, None otherwise
    """
    checker = UpdateChecker()
    return checker.check_for_updates()
