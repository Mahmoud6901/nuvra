"""Synthetic browser and native-app accessibility snapshots. No live desktop enumeration."""
import ctypes
import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from arabic_relay import meeting_detection as detection


class MeetingEvidence(TestCase):
    def test_browser_meeting_links_distinguish_a_room_from_landing_pages_and_similarly_named_sites(self):
        self.assertEqual(detection.page_app('https://meet.google.com/abc-defg-hij'), 'Google Meet')
        for address in (
            'https://meet.google.com/', 'https://meet.google.com.evil.example/abc-defg-hij',
            'https://example.org/?target=https://meet.google.com/abc-defg-hij',
            'https://zoom.us/pricing', 'https://teams.microsoft.com.evil.example/meetup-join/example',
            'https://www.google.com/search?q=meet.google.com',
        ):
            with self.subTest(address=address):
                self.assertFalse(detection.page_app(address))

    def test_microphone_activity_and_join_buttons_are_not_proof_of_a_call(self):
        for app in ('Zoom', 'Microsoft Teams', 'Google Meet'):
            for names in (
                ['Join now', 'Turn microphone on', 'Camera settings'],
                ['Waiting for the host', 'Unmute', 'Turn camera off'],
                ['Leave a comment', 'Participants in the study', 'Settings'],
                ['انضم الآن', 'اختبار الميكروفون', 'إعدادات الكاميرا'],
            ):
                with self.subTest(app=app, names=names):
                    self.assertFalse(detection.joined(app, names, []))

    def test_localized_post_join_controls_and_teams_ids_require_two_independent_signals(self):
        examples = [
            (detection.MEET, ['Leave call, Ctrl+Shift+H', 'Raise hand (Ctrl+Alt+H)'], [], 'Ask to join'),
            (detection.MEET, ['مُغادَرة المُكالَمة', 'رَفع اليَد'], [], 'طلب الانضمام'),
            (detection.ZOOM, ['Leave meeting (Alt+Q)', 'Participants'], [], 'Join meeting'),
            (detection.ZOOM, ['مغادرة الاجتماع', 'المشاركون'], [], 'الانضمام'),
        ]
        for app, controls, ids, lobby in examples:
            with self.subTest(app=app, controls=controls):
                self.assertTrue(detection.joined(app, controls, ids))
                self.assertFalse(detection.joined(app, controls[:1], ids))
                self.assertFalse(detection.joined(app, controls[1:], ids))
                self.assertFalse(detection.joined(app, [*controls, lobby], ids))
        self.assertTrue(detection.joined(detection.TEAMS, [], ['hangup-button', 'call-duration-custom']))
        for ids in (['hangup-button'], ['call-duration-custom'],
                    ['hangup-button', 'call-duration-custom', 'prejoin-join-button']):
            self.assertFalse(detection.joined(detection.TEAMS, [], ids))

    def test_supported_zoom_room_variants_and_teams_origins(self):
        for link in ('https://us02web.zoom.us/wc/123456789/join', 'https://zoom.us/wc/join/123456789'):
            self.assertEqual(detection.page_app(link), detection.ZOOM)
        for domain in ('teams.microsoft.com', 'teams.live.com', 'teams.cloud.microsoft'):
            self.assertEqual(detection.page_app('https://' + domain + '/v2/'), detection.TEAMS)
        for link in ('http://meet.google.com/abc-defg-hij', 'https://zoom.us/j/123456789',
                     'https://browser.test/wc/join/123456789', 'https://[invalid'):
            self.assertFalse(detection.page_app(link))


class CallLifecycle(TestCase):
    def setUp(self):
        self.windows, self.inspect = Mock(return_value=[]), Mock(return_value={})
        self.monitor = detection.Detector(self.windows, self.inspect)
        self.room = detection.Window(17, 30, 'chrome.exe', 'Chrome_WidgetWin_1', frozenset([detection.MEET]))
        self.tab = Mock()
        self.tab.state.return_value = 'selected'

    def test_background_call_stays_detected_until_the_tab_closes_or_its_selected_page_leaves(self):
        self.windows.return_value = [self.room]
        self.assertEqual(self.monitor.detect(), set())
        self.inspect.return_value = {detection.MEET: self.tab}
        self.assertEqual(self.monitor.detect(), {detection.MEET})
        self.tab.state.return_value = 'background'
        self.windows.return_value = [self.room._replace(apps=frozenset())]
        self.inspect.reset_mock()
        self.assertEqual(self.monitor.detect(), {detection.MEET})
        self.inspect.assert_not_called()
        self.tab.state.return_value = 'closed'
        self.assertEqual(self.monitor.detect(), set())
        self.windows.return_value = [self.room]
        self.tab.state.return_value = 'selected'
        self.assertEqual(self.monitor.detect(), {detection.MEET})
        self.inspect.return_value = {}
        self.assertEqual(self.monitor.detect(), set())

    def test_unreadable_window_only_preserves_a_previous_join_and_reused_handles_do_not_inherit_calls(self):
        self.inspect.return_value = None
        self.windows.return_value = [self.room]
        self.assertEqual(self.monitor.detect(), set())
        self.inspect.return_value = {detection.MEET: self.tab}
        self.assertEqual(self.monitor.detect(), {detection.MEET})
        self.inspect.return_value = None
        self.assertEqual(self.monitor.detect(), {detection.MEET})
        self.windows.return_value = [self.room._replace(pid=self.room.pid + 1)]
        self.assertEqual(self.monitor.detect(), set())

    def test_native_zoom_share_toolbar_keeps_a_call_when_the_previously_joined_window_is_hidden(self):
        window = detection.Window(42, 21, 'zoom.exe', 'ZPContentViewWndClass', frozenset([detection.ZOOM]))
        toolbar = window._replace(hwnd=43, cls='ZPFloatToolbarClass')
        self.windows.return_value = [window]
        self.inspect.return_value = {detection.ZOOM: None}
        self.assertEqual(self.monitor.detect(), {detection.ZOOM})
        self.inspect.return_value = {}
        self.windows.return_value = [window._replace(visible=False), toolbar]
        self.assertEqual(self.monitor.detect(), {detection.ZOOM})
        self.windows.return_value = [window._replace(visible=False)]
        self.assertEqual(self.monitor.detect(), set())


class AccessibilitySnapshots(TestCase):
    def setUp(self):
        uia = Mock()
        self.enterContext(patch.object(detection, 'automation', return_value=uia))
        self.enterContext(patch.dict(sys.modules, {'System': SimpleNamespace(IntPtr=int)}))
        self.enterContext(patch.object(ctypes, 'windll', SimpleNamespace(
            user32=SimpleNamespace(IsHungAppWindow=lambda hwnd: False)), create=True))
        self.controls = self.enterContext(patch.object(detection, 'read_controls'))

    def test_browser_reads_call_controls_only_from_the_supported_document(self):
        unrelated, room = Mock(), Mock()
        unrelated.GetCachedPropertyValue.return_value = 'https://example.org/article-about-google-meet'
        room.GetCachedPropertyValue.return_value = 'https://meet.google.com/abc-defg-hij'
        self.controls.return_value = (['Leave call', 'Raise hand'], [])
        chrome = detection.Window(91, 97, 'chrome.exe', 'Chrome_WidgetWin_1', frozenset([detection.MEET]))
        with patch.object(detection, 'find', return_value=[unrelated, room]), \
                patch.object(detection, 'selected_tab', return_value='joined-tab'):
            self.assertEqual(detection.inspect_window(chrome), {detection.MEET: 'joined-tab'})
        self.controls.assert_called_once()
        self.assertIs(self.controls.call_args.args[1], room)

    def test_native_lobby_overrides_a_previously_joined_window_even_if_minimized(self):
        services = [
            (detection.ZOOM, 'zoom.exe', 'ZPContentViewWndClass', (['Leave', 'Participants'], []),
             (['Join meeting', 'Test microphone'], [])),
            (detection.TEAMS, 'ms-teams.exe', 'TeamsWebView', ([], ['hangup-button', 'raisehands-button']),
             ([], ['prejoin-join-button'])),
        ]
        for name, exe, cls, joined, lobby in services:
            with self.subTest(service=name):
                window = detection.Window(3, 7, exe, cls, frozenset([name]))
                windows = Mock(return_value=[window])
                monitor = detection.Detector(windows, detection.inspect_window)
                self.controls.return_value = joined
                self.assertEqual(monitor.detect(), {name})
                windows.return_value = [window._replace(minimized=True)]
                self.controls.return_value = lobby
                self.assertEqual(monitor.detect(), set())
