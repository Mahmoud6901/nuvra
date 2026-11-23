"""Which supported calls has the user joined on this PC? Windows only.

Win32 picks out the few windows that could hold a call. UI Automation then looks for the controls that exist only after
joining. Window titles, page addresses and control names are compared in memory and dropped: nothing is stored or logged,
no device is opened and nothing is sent anywhere.

Browsers expose only the selected tab, so a call is first noticed while its tab is selected (the window may be behind
others or minimized). From then on it stays detected, whatever tab is in front, until that tab closes or shows that the
call ended.
"""
import functools
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

MEET, ZOOM, TEAMS = 'Google Meet', 'Zoom', 'Microsoft Teams'

BROWSERS = {'chrome.exe', 'msedge.exe', 'brave.exe', 'vivaldi.exe', 'opera.exe', 'chromium.exe', 'arc.exe', 'aside.exe',
            'comet.exe', 'firefox.exe'}
BROWSER_CLASSES = {'Chrome_WidgetWin_1', 'MozillaWindowClass'}
# Zoom's home, sign-in, preview and waiting windows use other classes.
ZOOM_CLASSES = {'ZPContentViewWndClass', 'ConfMultiTabContentWndClass'}
ZOOM_FLOATING = {'ZPFloatToolbarClass', 'ZPFloatVideoWndClass'}   # stand in for a hidden meeting window
# A browser page is inspected only when its window title could be a call. The page address decides.
TITLE_HINTS = {'meet': MEET, 'zoom': ZOOM, 'teams': TEAMS}


class Window(NamedTuple):
    hwnd: int
    pid: int
    process: str           # lower-case file name
    cls: str
    apps: frozenset        # calls this window could hold; empty for a browser showing something else
    visible: bool = True   # minimized windows are visible
    minimized: bool = False

    @property
    def browser(self):
        return self.process in BROWSERS


def fold(name):
    """The comparable part of an accessible name: no shortcut hint, direction marks, diacritics or letter variants."""
    name = re.sub(r'[(\[].*', '', unicodedata.normalize('NFKC', name or ''))
    name = ''.join(c for c in name if unicodedata.category(c) not in ('Cf', 'Mn'))
    name = re.split('[,،]', name.translate(str.maketrans('أإآى', 'اااي')))[0]
    return ' '.join(name.casefold().split())


def names(*values):
    return frozenset(map(fold, values))


# A leave control alone also shows while waiting to be admitted, so each rule needs a control for taking part as well.
# Microphone, camera, device and "more" controls never count: preview screens have them.
LEAVE = {
    MEET: names('Leave call', 'Leave meeting', 'مغادرة المكالمة', 'مغادرة الاجتماع', 'الخروج من المكالمة', 'إنهاء المكالمة'),
    ZOOM: names('Leave', 'End', 'Leave meeting', 'End meeting', 'Stop share', 'Stop sharing',
                'مغادرة', 'إنهاء', 'مغادرة الاجتماع', 'إنهاء الاجتماع', 'إيقاف المشاركة'),
}
TAKE_PART = {
    MEET: names('Raise hand', 'Lower hand', 'Send a reaction', 'Chat with everyone', 'Show everyone', 'People',
                'رفع اليد', 'خفض اليد', 'رفع اليد أو خفضها', 'إرسال تفاعل', 'الدردشة مع الجميع', 'الأشخاص',
                'المستخدمون', 'عرض الجميع', 'إظهار الجميع'),
    ZOOM: names('Participants', 'Manage participants', 'Open the participants list', 'Close the participants list',
                'Open the participants list pane', 'Close the participants list pane',
                'Open the manage participants list pane', 'Close the manage participants list pane',
                'Share', 'Share screen', 'New share', 'Start share', 'Raise hand', 'Lower hand', 'Reactions', 'React',
                'المشاركون', 'إدارة المشاركين', 'مشاركة الشاشة', 'مشاركة جديدة', 'رفع اليد', 'خفض اليد', 'التفاعلات'),
}
NOT_JOINED = {
    MEET: names('Join now', 'Ask to join', 'Join anyway', 'Join here too', 'Rejoin',
                'الانضمام الآن', 'طلب الانضمام', 'إعادة الانضمام'),
    ZOOM: names('Join', 'Join meeting', 'انضمام', 'الانضمام', 'الانضمام إلى الاجتماع'),
}
# Teams keeps the same element ids in every language.
TEAMS_LEAVE = 'hangup-button'
TEAMS_TAKE_PART = {'raisehands-button', 'reaction-menu-button', 'call-duration-custom'}
TEAMS_NOT_JOINED = 'prejoin-join-button'
TEAMS_IDS = {TEAMS_LEAVE, TEAMS_NOT_JOINED, *TEAMS_TAKE_PART}


def joined(app, labels=(), ids=()):
    """Do these enabled controls belong to a call the user is in? Teams is judged by ids, the others by button names."""
    if app == TEAMS:
        ids = set(ids)
        return TEAMS_LEAVE in ids and TEAMS_NOT_JOINED not in ids and bool(ids & TEAMS_TAKE_PART)
    labels = set(map(fold, labels))
    return bool(labels & LEAVE[app] and labels & TAKE_PART[app]) and not labels & NOT_JOINED[app]


def not_joined(app, labels=(), ids=()):
    """Do these controls show outright that the user is not in the call? Missing call controls alone do not."""
    return TEAMS_NOT_JOINED in ids if app == TEAMS else not NOT_JOINED[app].isdisjoint(map(fold, labels))


def page_app(url):
    """The supported call page at this address, or None for any other page, including each service's home page."""
    try:
        parts = urlsplit(url or '')
        host = (parts.hostname or '').lower()
    except ValueError:
        return None
    if parts.scheme != 'https':
        return None
    if host == 'meet.google.com':
        return MEET if re.match(r'/(?:[a-z]{3}-[a-z]{4}-[a-z]{3}|lookup/[^/]+)/?$', parts.path) else None
    if host == 'zoom.us' or host.endswith('.zoom.us'):
        return ZOOM if re.match(r'/wc/(?:join/)?\d+(?:/|$)', parts.path) else None
    return TEAMS if host in ('teams.microsoft.com', 'teams.live.com', 'teams.cloud.microsoft') else None


class Detector:
    """Remembers where each joined call is, so tab switches and hidden toolbars do not look like leaving.

    windows() lists Window values. inspect(window) returns {app: tab} for the calls it shows, with False as the tab when
    the window shows a preview, lobby or ended call, or returns None when the window cannot be read right now.
    A tab is None or has state() -> 'selected', 'background' or 'closed'.
    """

    def __init__(self, windows, inspect):
        self.windows, self.inspect = windows, inspect
        self.calls = {}   # (hwnd, pid) -> {app: tab}

    def detect(self):
        windows = list(self.windows())
        calls = {}
        for window in windows:
            key = window.hwnd, window.pid
            found = self.look(window, self.calls.get(key, {}), windows)
            if found:
                calls[key] = found
        self.calls = calls
        return {app for found in calls.values() for app in found}

    def look(self, window, known, windows):
        if not window.visible:
            # Zoom hides the meeting window behind its floating toolbar while sharing.
            return known if any(w.visible and w.pid == window.pid and w.cls in ZOOM_FLOATING for w in windows) else {}
        tabs = {app: tab.state() if tab is not None else 'selected' for app, tab in known.items()}
        kept = {app: known[app] for app, state in tabs.items() if state == 'background'}
        current = {app: known[app] for app, state in tabs.items() if state == 'selected'}
        if not window.apps and not current:
            return kept
        found = self.inspect(window)
        if found is None or not window.browser and (window.minimized or ZOOM in current):
            # Unreadable, minimized, or a Zoom toolbar that hid itself: the window still being there is what counts,
            # unless it shows outright that the call is over.
            found = {**current, **(found or {})}
        return {app: tab for app, tab in {**kept, **found}.items() if tab is not False}


# ---- Windows ----

def list_windows():
    """Top-level windows that could hold a call, in front or not. Titles never leave this function."""
    import ctypes
    import os
    from ctypes import wintypes as w
    user, kernel = ctypes.WinDLL('user32'), ctypes.WinDLL('kernel32')
    callback = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    for library, name, result, arguments in (
            (user, 'EnumWindows', w.BOOL, [callback, w.LPARAM]), (user, 'IsWindowVisible', w.BOOL, [w.HWND]),
            (user, 'IsIconic', w.BOOL, [w.HWND]), (user, 'GetClassNameW', ctypes.c_int, [w.HWND, w.LPWSTR, ctypes.c_int]),
            (user, 'GetWindowTextW', ctypes.c_int, [w.HWND, w.LPWSTR, ctypes.c_int]),
            (user, 'GetWindowThreadProcessId', w.DWORD, [w.HWND, ctypes.POINTER(w.DWORD)]),
            (kernel, 'OpenProcess', w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]), (kernel, 'CloseHandle', w.BOOL, [w.HANDLE]),
            (kernel, 'QueryFullProcessImageNameW', w.BOOL, [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)])):
        function = getattr(library, name)
        function.restype, function.argtypes = result, arguments
    processes, found = {os.getpid(): ''}, []

    def process_name(pid):
        if pid not in processes:
            processes[pid] = ''
            handle = kernel.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                path, size = ctypes.create_unicode_buffer(1024), w.DWORD(1024)
                if kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                    processes[pid] = path.value.rpartition('\\')[2].lower()
                kernel.CloseHandle(handle)
        return processes[pid]

    @callback
    def visit(hwnd, _):
        buffer = ctypes.create_unicode_buffer(512)
        user.GetClassNameW(hwnd, buffer, 256)
        cls, visible = buffer.value, bool(user.IsWindowVisible(hwnd))
        zoom = cls in ZOOM_CLASSES or cls in ZOOM_FLOATING
        if not zoom and not (visible and (cls == 'TeamsWebView' or cls in BROWSER_CLASSES)):
            return True
        pid = w.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process = process_name(pid.value)
        if zoom and process == 'zoom.exe':
            apps = {ZOOM}
        elif cls == 'TeamsWebView' and process == 'ms-teams.exe' or process == 'teams.exe':
            apps = {TEAMS}
        elif cls in BROWSER_CLASSES and process in BROWSERS:
            user.GetWindowTextW(hwnd, buffer, 512)
            title = buffer.value.casefold()
            apps = {app for hint, app in TITLE_HINTS.items() if hint in title}
        else:
            return True
        found.append(Window(hwnd, pid.value, process, cls, frozenset(apps), visible, bool(user.IsIconic(hwnd))))
        return True

    user.EnumWindows(visit, 0)
    return found


@functools.cache
def automation():
    """The .NET UI Automation client that ships with Windows, or None where it cannot load."""
    try:
        import clr
        from System.Runtime.InteropServices import RuntimeEnvironment
        folder = str(Path(str(RuntimeEnvironment.GetRuntimeDirectory())) / 'WPF')
        if folder not in sys.path:
            sys.path.append(folder)
        clr.AddReference('UIAutomationClient')
        clr.AddReference('UIAutomationTypes')
        import System.Windows.Automation as uia
        uia.AutomationElement.NameProperty
        return uia
    except Exception:
        logging.warning('Meeting detection is off: Windows UI Automation could not be loaded')
        return None


class Tab:
    """A browser tab, followed without reading its title or its page."""

    def __init__(self, element):
        self.element = element

    def state(self):
        uia = automation()
        try:
            # Reading this is what fails once a tab closes; the selection alone falls back to a default.
            if self.element.GetCurrentPropertyValue(uia.AutomationElement.ControlTypeProperty) != uia.ControlType.TabItem:
                return 'closed'
            selected = self.element.GetCurrentPropertyValue(uia.SelectionItemPattern.IsSelectedProperty)
        except uia.ElementNotAvailableException:
            return 'closed'
        except Exception:   # a busy browser; the window disappears from the list if it is gone
            return 'background'
        return 'selected' if selected is True else 'background'


def either(uia, conditions):
    from System import Array
    conditions = list(conditions)
    return conditions[0] if len(conditions) == 1 else uia.OrCondition(Array[uia.Condition](conditions))


def find(uia, root, condition, *properties):
    """Matching descendants with these properties fetched in the same call."""
    request = uia.CacheRequest()
    for value in properties:
        request.Add(value)
    active = request.Activate()   # pythonnet's IDisposable is not a context manager
    try:
        return list(root.FindAll(uia.TreeScope.Descendants, condition))
    finally:
        active.Dispose()


def read_controls(uia, root, app):
    """(button names, element ids) of the controls under root. Only ids are read for Teams.

    A disabled control is no proof of a call, but a disabled join button still shows that there is none.
    """
    element, kind = uia.AutomationElement, uia.ControlType
    wanted = either(uia, [uia.PropertyCondition(element.AutomationIdProperty, value) for value in sorted(TEAMS_IDS)]
                    if app == TEAMS else
                    [uia.PropertyCondition(element.ControlTypeProperty, value)
                     for value in (kind.Button, kind.SplitButton, kind.MenuItem)])
    value = element.AutomationIdProperty if app == TEAMS else element.NameProperty
    read = []
    for e in find(uia, root, wanted, value, element.IsEnabledProperty):
        text = str(e.GetCachedPropertyValue(value) or '')
        if e.GetCachedPropertyValue(element.IsEnabledProperty) or not_joined(app, [text], [text]):
            read.append(text)
    return ([], read) if app == TEAMS else (read, [])


def judge(calls, app, controls):
    if joined(app, *controls):
        calls[app] = None
    elif not_joined(app, *controls):
        calls.setdefault(app, False)


def selected_tab(uia, root):
    """The browser's own selected tab; pages can contain tabs too."""
    element, walker = uia.AutomationElement, uia.TreeWalker.ControlViewWalker
    condition = uia.AndCondition(uia.PropertyCondition(element.ControlTypeProperty, uia.ControlType.TabItem),
                                 uia.PropertyCondition(uia.SelectionItemPattern.IsSelectedProperty, True))
    for tab in root.FindAll(uia.TreeScope.Descendants, condition):
        parent = walker.GetParent(tab)
        while parent is not None and not element.op_Equality(parent, root):
            if parent.Current.ControlType == uia.ControlType.Document:
                break
            parent = walker.GetParent(parent)
        else:
            return Tab(tab)
    return None


def inspect_window(window):
    uia = automation()
    if uia is None:
        return {}
    try:
        import ctypes
        from System import IntPtr
        if ctypes.windll.user32.IsHungAppWindow(ctypes.c_void_p(window.hwnd)):
            return None
        root = uia.AutomationElement.FromHandle(IntPtr(window.hwnd))
        if not window.browser:
            calls = {}
            for app in window.apps:
                judge(calls, app, read_controls(uia, root, app))
            return calls
        address = uia.ValuePattern.ValueProperty
        pages = find(uia, root, uia.PropertyCondition(uia.AutomationElement.ControlTypeProperty, uia.ControlType.Document),
                     address)
        read, calls = False, {}
        for page in pages:
            url = page.GetCachedPropertyValue(address)
            app = page_app(url) if isinstance(url, str) else None
            if app is None:
                read = read or bool(isinstance(url, str) and url)
                continue
            labels, ids = read_controls(uia, page, app)
            if app != TEAMS and not labels:   # the browser has not filled in the page yet
                continue
            read = True
            judge(calls, app, (labels, ids))
        if None in calls.values():
            tab = selected_tab(uia, root)
            calls = {app: tab if found is None else found for app, found in calls.items()}
        return calls if read else None
    except Exception:
        # The window closed or changed while it was being read. No details: they can carry titles and addresses.
        return None


_detector = Detector(list_windows, inspect_window)


def detect():
    """Labels of the calls joined on this PC right now. Call from one thread, and not from a UI thread."""
    return _detector.detect() if sys.platform == 'win32' else set()
