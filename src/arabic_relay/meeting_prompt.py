"""Ask once when a supported call is detected. Only a click on the current offer can start recording."""
import logging
import queue
import secrets
import threading
import time

from . import store


def enabled():
    return bool(store.get_setting('meeting_prompts', 0))


def set_enabled(value):
    if type(value) is not bool:
        raise ValueError('Choose whether to show meeting prompts.')
    store.set_setting('meeting_prompts', int(value))
    return {'enabled': value}


class PromptMonitor:
    """The monitor thread owns detection, the popup and its decisions. UI callbacks only post button events."""

    def __init__(self, enabled, available, detect, show, record):
        self.enabled, self.available, self.detect = enabled, available, detect
        self.show, self.record = show, record
        self.handled = {}
        self.app = self.token = self.window = None
        self.actions = queue.Queue()
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is None and not self.stopped.is_set():
            self.thread = threading.Thread(target=self.run, name='meeting-prompts', daemon=True)
            self.thread.start()

    def action(self, name, token):
        if not self.stopped.is_set():
            self.actions.put((name, token))

    def close(self):
        window = self.window
        self.window = self.app = self.token = None
        if window is not None:
            window.close()

    def poll(self, action=None):
        if self.stopped.is_set() or not self.enabled():
            self.handled.clear()
            self.close()
            return
        if action == ('dismiss', self.token) and self.window is not None:
            self.close()
            return
        calls = self.detect()
        now = time.monotonic()
        for call, last in list(self.handled.items()):
            if call in calls:
                self.handled[call] = now
            elif now - last >= 20:
                del self.handled[call]
        if not calls and self.window is None:
            return
        if not self.available() or self.stopped.is_set():
            self.handled.update(dict.fromkeys(calls, now))
            self.close()
            return
        if self.window is not None:
            if self.app not in self.handled:
                self.close()
            elif action == ('record', self.token):
                if self.app not in calls or not self.enabled() or self.stopped.is_set():
                    self.close()
                else:
                    try:
                        self.record()
                    except Exception:
                        self.window.failed()
                    else:
                        self.close()
                return
        if self.window is None:
            for app in sorted(calls):
                if app in self.handled:
                    continue
                if self.stopped.is_set() or not self.enabled():
                    return
                token = secrets.token_urlsafe(18)
                window = self.show(app, token)
                self.window, self.app, self.token = window, app, token
                self.handled[app] = now
                break

    def run(self):
        action = None
        try:
            while not self.stopped.is_set():
                try:
                    self.poll(action)
                except Exception:
                    logging.debug('Meeting detection temporarily unavailable', exc_info=False)
                    if action and action[0] == 'record' and self.window is not None:
                        self.window.failed()
                try:
                    action = self.actions.get(timeout=4)
                except queue.Empty:
                    action = None
        finally:
            self.close()

    def stop(self):
        self.stopped.set()
        self.actions.put(None)
