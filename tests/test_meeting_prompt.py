from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import arabic_relay
from arabic_relay import desktop, meeting_prompt, paths


class JoinMonitor(TestCase):
    def setUp(self):
        self.enabled = Mock(return_value=True)
        self.available = Mock(return_value=True)
        self.detect = Mock(return_value=set())
        self.show = Mock(side_effect=lambda app, token: Mock())
        self.record = Mock()
        self.monitor = meeting_prompt.PromptMonitor(self.enabled, self.available, self.detect, self.show, self.record)
        self.clock = self.enterContext(patch.object(meeting_prompt.time, 'monotonic', return_value=100))

    def poll(self, when, calls, action=None):
        self.clock.return_value = when
        self.detect.return_value = set(calls)
        self.monitor.poll(action)

    def test_dismiss_once_and_rearm_only_after_leaving_then_rejoining(self):
        self.poll(100, ['Zoom'])
        first = self.monitor.token
        window = self.monitor.window
        self.poll(101, ['Zoom'], ('dismiss', first))
        window.close.assert_called_once()
        self.poll(115, ['Zoom'])
        self.poll(120, [])
        self.poll(125, ['Zoom'])
        self.show.assert_called_once()
        self.poll(146, [])
        self.poll(150, ['Zoom'])
        self.assertEqual(self.show.call_count, 2)
        self.assertNotEqual(self.monitor.token, first)
        self.poll(151, ['Zoom'], ('record', first))
        self.poll(152, ['Zoom'], ('dismiss', first))
        self.assertIsNotNone(self.monitor.window)
        self.record.assert_not_called()

    def test_explicit_current_record_click_starts_once_and_a_replayed_click_is_ignored(self):
        self.poll(100, ['Microsoft Teams'])
        token = self.monitor.token
        self.record.assert_not_called()
        self.poll(101, ['Microsoft Teams'], ('record', token))
        self.poll(102, ['Microsoft Teams'], ('record', token))
        self.record.assert_called_once()
        self.show.assert_called_once()
        self.assertIsNone(self.monitor.window)

    def test_active_recording_disabled_preference_or_call_ending_invalidates_offer(self):
        for condition in ('active', 'disabled', 'ended'):
            with self.subTest(condition=condition):
                self.enabled.return_value = self.available.return_value = True
                self.monitor.handled.clear()
                self.poll(100, ['Google Meet'])
                token = self.monitor.token
                self.available.return_value = condition != 'active'
                self.enabled.return_value = condition != 'disabled'
                self.poll(105, [] if condition == 'ended' else ['Google Meet'], ('record', token))
                self.record.assert_not_called()
                self.assertIsNone(self.monitor.window)

    def test_disabled_monitor_never_queries_desktop_and_an_active_recorder_suppresses_until_rejoin(self):
        self.enabled.return_value = False
        self.poll(100, ['Zoom'])
        self.detect.assert_not_called()
        self.enabled.return_value = True
        self.available.return_value = False
        self.poll(101, ['Zoom'])
        self.available.return_value = True
        self.poll(105, ['Zoom'])
        self.show.assert_not_called()
        self.poll(129, [])
        self.poll(130, ['Zoom'])
        self.show.assert_called_once()

    def test_failed_start_allows_retry_and_dismissal_without_starting_again(self):
        self.poll(100, ['Zoom'])
        token, window = self.monitor.token, self.monitor.window
        self.record.side_effect = ValueError('recorder unavailable')
        self.poll(101, ['Zoom'], ('record', token))
        window.failed.assert_called_once()
        self.assertIs(self.monitor.window, window)
        self.poll(102, ['Zoom'], ('dismiss', token))
        window.close.assert_called_once()
        self.record.assert_called_once()

    def test_shutdown_during_slow_availability_check_cannot_start_capture(self):
        self.poll(100, ['Zoom'])
        token = self.monitor.token

        def close_app():
            self.monitor.stop()
            return True

        self.available.side_effect = close_app
        self.poll(101, ['Zoom'], ('record', token))
        self.record.assert_not_called()
        self.assertIsNone(self.monitor.window)

    def test_preference_turned_off_during_detection_cannot_open_a_new_popup(self):
        def call_appears_after_turning_off():
            self.enabled.return_value = False
            return {'Google Meet'}

        self.detect.side_effect = call_appears_after_turning_off
        self.monitor.poll()
        self.show.assert_not_called()
        self.record.assert_not_called()


class PopupPreferences(TestCase):
    def test_opt_in_is_persisted_only_in_current_profile_and_requires_a_boolean(self):
        with TemporaryDirectory() as temp, patch.object(paths, 'DATA', Path(temp)):
            self.assertFalse(meeting_prompt.enabled())
            for value in ['true', None, 1]:
                with self.assertRaises(ValueError):
                    meeting_prompt.set_enabled(value)
            self.assertFalse(meeting_prompt.enabled())
            self.assertTrue(meeting_prompt.set_enabled(True)['enabled'])
            self.assertTrue(meeting_prompt.enabled())
            self.assertFalse(meeting_prompt.set_enabled(False)['enabled'])

    def test_native_popup_accepts_only_its_own_token_and_named_actions(self):
        forward = Mock()
        api = desktop.MeetingPromptApi(forward, 'current-window')
        for action, token in [('record', 'old-window'), ('unknown', 'current-window')]:
            with self.assertRaises(ValueError):
                api.choose(action, token)
        forward.assert_not_called()
        api.choose('dismiss', 'current-window')
        forward.assert_called_once_with('dismiss', 'current-window')

    def test_native_offer_is_topmost_without_focus_and_close_maps_only_to_dismiss(self):
        display = MagicMock()
        display.screens = [SimpleNamespace(x=0, y=0, width=1920, height=1080)]
        window = display.create_window.return_value
        closed = window.events.closed
        display.windows = [window]
        user32 = SimpleNamespace(GetDpiForSystem=lambda: 96)
        send = Mock()
        with patch.object(desktop, 'webview', display), \
                patch.object(desktop.ctypes, 'windll', SimpleNamespace(user32=user32), create=True):
            popup = desktop.MeetingPromptWindow('Microsoft Teams', 'one-time-window', send)
            args, options = display.create_window.call_args
            link = urlsplit(args[1])
            self.assertEqual(link.path, '/meeting-prompt')
            self.assertEqual(parse_qs(link.fragment), {'app': ['Microsoft Teams'], 'token': ['one-time-window']})
            self.assertTrue(options['on_top'])
            self.assertFalse(options['focus'])
            send.assert_not_called()
            on_close, = closed.__iadd__.call_args.args
            on_close()
            send.assert_called_once_with('dismiss', 'one-time-window')
            popup.close()
            window.destroy.assert_called_once()

    def test_monitor_wires_an_explicit_record_choice_to_microphone_and_speakers(self):
        detector = SimpleNamespace(detect=Mock(return_value={'Zoom'}))
        with patch.object(arabic_relay, 'meeting_detection', detector, create=True), \
                patch.object(desktop.meeting_prompt, 'enabled', return_value=True), \
                patch.object(desktop.meetings, 'status', return_value={'available': True, 'state': 'idle'}), \
                patch.object(desktop.meetings, 'active', return_value=False), \
                patch.object(desktop, 'MeetingPromptWindow') as window, \
                patch.object(desktop.server, 'start_meeting') as start, \
                patch.object(desktop.webview, 'windows', []):
            monitor = desktop.make_meeting_monitor()
            monitor.poll()
            window.assert_called_once()
            start.assert_not_called()
            monitor.poll(('record', monitor.token))
            start.assert_called_once_with(microphone=True, speakers=True)
