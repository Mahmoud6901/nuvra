"""Find outside tools and the Whisper model: the usual places first, then what the user picked."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import sys
if os.name == 'nt':
    import winreg
else:
    winreg = None

from . import paths, store

HF_REPOS = ('models--Systran--faster-whisper-large-v3', 'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo',
            'models--Abdallah151--faster-whisper-egyptian-code-switching-large-v3-turbo')
APP_MODELS = ('large-v3', 'large-v3-turbo', 'large-v3-turbo-egyptian')  # downloaded into paths.DATA/models
TOOLS = ('codex', 'claude', 'wispr')
UNINSTALL = r'Software\Microsoft\Windows\CurrentVersion\Uninstall'
FORMAT_HELP = 'Nuvra محتاج نسخة faster-whisper (Systran/faster-whisper-large-v3).'


def validate_model_folder(folder):
    folder = Path(folder)
    snapshots = folder / 'snapshots'
    if snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                return validate_model_folder(snapshot)
            except ValueError:
                continue
    if folder.is_dir():
        if any(folder.glob('*.pt')):
            raise ValueError('ده نموذج OpenAI Whisper بصيغة pt. ' + FORMAT_HELP)
        if any(folder.glob('ggml*.bin')):
            raise ValueError('ده نموذج whisper.cpp. ' + FORMAT_HELP)
        if all((folder / name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json')) and (
                (folder / 'vocabulary.json').is_file() or (folder / 'vocabulary.txt').is_file()):
            return folder
    raise ValueError("الفولدر ده مش نموذج faster-whisper. اختار الفولدر اللي فيه model.bin و config.json.")


def hf_cache():
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE'])
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']) / 'hub'
    return Path.home() / '.cache/huggingface/hub'


def whisper_model(saved):
    found = [paths.DATA / 'models' / name for name in APP_MODELS] + [hf_cache() / repo for repo in HF_REPOS]
    for folder in ([saved] if saved else []) + found:
        try:
            return validate_model_folder(folder)
        except (ValueError, OSError):
            continue
    return None


def translator_path():
    if sys.platform == 'darwin':
        return unix_tool('codex')
    # Use the native executable; npm's .cmd shim requires a shell on Windows.
    installed = shutil.which('codex.exe')
    npm = Path(os.environ.get('APPDATA', '')) / 'npm/node_modules/@openai/codex/node_modules'
    candidates = [npm / '@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe',
                  Path.home() / '.local/bin/codex.exe', winget_links() / 'codex.exe']
    # Codex desktop updates replace this versioned directory; discover it instead of pinning a version.
    desktop = Path(os.environ.get('LOCALAPPDATA', '')) / 'OpenAI/Codex/bin'
    candidates += sorted(desktop.glob('*/codex.exe'), key=lambda p: p.stat().st_mtime, reverse=True)
    return installed or next((str(p) for p in candidates if p.is_file()), None)


def claude_path():
    if sys.platform == 'darwin':
        return unix_tool('claude')
    candidates = [Path.home() / '.local/bin/claude.exe', winget_links() / 'claude.exe']
    return shutil.which('claude.exe') or next((str(p) for p in candidates if p.is_file()), None)


def wispr_flow_path():
    if sys.platform == 'darwin':
        roots = [Path('/Applications'), Path.home() / 'Applications']
        return next((p for root in roots for p in (root / 'Wispr Flow.app',)
                     if p.is_dir()), None)
    roots = [Path(os.environ.get('LOCALAPPDATA', '')) / 'WisprFlow',
             Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Wispr Flow']
    roots += [Path(os.environ[name]) / 'Wispr Flow' for name in ('ProgramFiles', 'ProgramFiles(x86)') if os.environ.get(name)]
    roots += registry_install_folders('Wispr Flow')  # custom install folders
    for root in roots:
        candidates = [root / 'Wispr Flow.exe', root / 'WisprFlow.exe']
        candidates += sorted(root.glob('app-*/Wispr Flow.exe'), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
                return candidate
    return None


def winget_links():
    return Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Links'


def registry_install_folders(name):
    """InstallLocation of every installed program whose DisplayName is `name` (Windows "Installed apps")."""
    folders = []
    if winreg is None:
        return folders
    for hive, key_path in ((winreg.HKEY_CURRENT_USER, UNINSTALL), (winreg.HKEY_LOCAL_MACHINE, UNINSTALL),
                           (winreg.HKEY_LOCAL_MACHINE, UNINSTALL.replace('Software', 'Software\\WOW6432Node', 1))):
        try:
            root = winreg.OpenKey(hive, key_path)
        except OSError:
            continue
        index = 0
        while True:
            try:
                entry = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                key = winreg.OpenKey(root, entry)
                if winreg.QueryValueEx(key, 'DisplayName')[0] == name:
                    location = winreg.QueryValueEx(key, 'InstallLocation')[0]
                    if isinstance(location, str) and location.strip():
                        folders.append(Path(location.strip()))
            except OSError:
                continue
    return folders


AUTO_DETECT = {'codex': 'translator_path', 'claude': 'claude_path', 'wispr': 'wispr_flow_path'}


def unix_tool(name):
    # Finder launches do not inherit the user's shell PATH.
    candidates = [Path.home() / '.local/bin' / name, Path('/opt/homebrew/bin') / name,
                  Path('/usr/local/bin') / name]
    return shutil.which(name) or next((str(p) for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)


def valid_tool(program, kind):
    if sys.platform == 'darwin':
        if kind == 'wispr':
            return program.suffix == '.app' and (program / 'Contents/Info.plist').is_file()
        return program.is_file() and os.access(program, os.X_OK)
    return program.is_file() and program.suffix.lower() == '.exe'


def check_kind(kind):
    if kind not in TOOLS:
        raise ValueError('اختار Codex أو Claude Code أو Wispr Flow.')


def tool_path(kind):
    """The program the user chose with Browse… while it still exists, else the one found in the usual places."""
    check_kind(kind)
    saved = store.get_setting(kind + '_path')
    if saved and valid_tool(Path(saved), kind):
        return Path(saved)
    found = globals()[AUTO_DETECT[kind]]()
    return Path(found) if found else None


def set_tool_path(kind, path):
    check_kind(kind)
    if not isinstance(path, str) or len(path) > 1000:
        raise ValueError('اختار ملف البرنامج بصيغة exe.')
    if not path.strip():
        store.set_setting(kind + '_path', '')
        return {'kind': kind, 'path': None}
    program = Path(path.strip())
    if not valid_tool(program, kind):
        raise ValueError('Choose an executable CLI or the Wispr Flow application.' if sys.platform == 'darwin'
                         else 'اختار ملف البرنامج بصيغة exe.')
    store.set_setting(kind + '_path', str(program))
    return {'kind': kind, 'path': str(program)}


def missing_tools():
    """Browse… choices whose program was moved or deleted since."""
    return {kind: saved for kind in TOOLS if (saved := store.get_setting(kind + '_path')) and not valid_tool(Path(saved), kind)}


@contextmanager
def external_runtime():
    """External programs must not inherit a frozen app's Python DLL directory."""
    frozen = os.name == 'nt' and getattr(sys, 'frozen', False)
    if frozen:
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        yield
    finally:
        if frozen:
            ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)
