# Nuvra

Speak Arabic and English. Paste clear English text into the app where you are working.

Nuvra is an open-source dictation app for Windows, with experimental macOS builds. Start recording with a keyboard shortcut, speak naturally, and finish to paste the result into your active text field.

[Website](https://mahmoud6901.github.io/nuvra/) · [Download for Windows](https://github.com/Mahmoud6901/nuvra/releases/latest) · [macOS preview](#macos-experimental) · [Report a problem](https://github.com/Mahmoud6901/nuvra/issues/new?template=bug_report.yml)

## Install and get started

### Windows

1. Download the Windows installer from the [website](https://mahmoud6901.github.io/nuvra/#download) and run it. New builds are named `Nuvra-Setup-<version>.exe`. Installation is per user.
2. Open Nuvra and choose a speech model in **Models**. Local Whisper models download from this tab. Cohere and Wispr Flow are also available.
3. Install and sign in to Codex CLI or Claude Code, then select your preferred model in **Models** for English prompt preparation and translation fallback.
4. Choose **Translate Arabic only** or **Structured prompt** in **Settings**.
5. Keep Nuvra running, click in the destination text field, and press **Ctrl + Alt + Space**. Speak, then press the shortcut again to finish and paste.

Nuvra requires Windows 10 or 11, 64-bit, and Microsoft Edge WebView2. The installer installs WebView2 if needed. Local speech models require a one-time download and several gigabytes of free disk space. An NVIDIA GPU is optional. CPU processing is supported.

### macOS (experimental)

macOS 15 or newer is required. Download the Mac preview from the [website](https://mahmoud6901.github.io/nuvra/#macos-download) or the [release assets](https://github.com/Mahmoud6901/nuvra/releases/latest). If a release has no Mac assets, use the artifacts from a successful [macOS builds workflow](https://github.com/Mahmoud6901/nuvra/actions/workflows/macos.yml?query=branch%3Amain+is%3Asuccess) or [build from source](#build-macos-release-assets). Downloading workflow artifacts requires a GitHub sign-in.

Download `Nuvra-<version>-macos-arm64.dmg` for Apple Silicon or `Nuvra-<version>-macos-x86_64.dmg` for Intel from the release assets. Open the DMG and drag **Nuvra** into **Applications**. ZIP archives of the same app are also available. Verify downloads against the matching `SHA256SUMS-macos-<architecture>.txt` file.

These builds have an ad-hoc signature, not an Apple Developer ID signature or notarization. macOS may block the first launch. After attempting to open the app, use **System Settings → Privacy & Security → Open Anyway** if you trust the download. This does not require disabling Gatekeeper.

Allow microphone access when prompted. For global shortcuts and automatic paste, enable **Nuvra** under **System Settings → Privacy & Security → Accessibility** (called **Device Control and Data Access** on some macOS versions), then quit and reopen it. If the shortcut status also requests **Input Monitoring**, enable that permission and restart.

Choose your speech model, prompt provider, and writing mode as described above. Keep Nuvra running and click in the destination text field before recording. The default shortcut is **Control + Option + Space**. Press it again or press plain **Return** to finish recording; **Escape** cancels while the recording pill is open.

Local Whisper runs on the CPU on macOS; Metal and CUDA acceleration are not included. Automatic paste requires the destination app to expose its focused text field through Accessibility. If it cannot verify that field, Nuvra preserves the text for manual copying. macOS updates are notification-only: download the new DMG, quit Nuvra, and replace the app in Applications.

Codex and Claude Code are discovered on `PATH`, in `~/.local/bin`, and in the usual Homebrew directories. Wispr Flow is detected in `/Applications` or `~/Applications`.

Saved service credentials are encrypted with a key stored in your macOS login Keychain. If the Keychain is locked or access is denied, Nuvra reports an error instead of saving plaintext credentials.

## Writing modes

| Mode | Result |
| --- | --- |
| **Translate Arabic only** | Translates Arabic phrases while preserving the surrounding English wording. |
| **Structured prompt** | Turns mixed Arabic and English speech into organized English instructions for an AI assistant. |

For example, you could say “راجع صفحة الـ login وحسّن الـ error messages” and receive “Review the login page and improve the error messages.”

For local Arabic translation, install Ollama and download `gemma4:e4b` from **Models**. If the local translator is unavailable or cannot produce a usable answer, Nuvra uses your selected Codex or Claude Code provider.

## Recording controls

A small recording pill appears while you speak. Finishing processes the text and pastes it into the original app.

| Action | Windows | macOS |
| --- | --- | --- |
| Start or finish recording | **Ctrl + Alt + Space** | **Control + Option + Space** |
| Finish recording and paste | **Enter** or **✓** in the pill | **Return** or **✓** in the pill |
| Cancel recording | **Esc** or **✕** in the pill | **Escape** or **✕** in the pill |
| Paste copied text manually | **Ctrl + V** | **Command + V** |

Choose a different recording shortcut in **Settings**. Windows also offers **Ctrl + Alt + R** and **Ctrl + Shift + Space**; macOS offers **Control + Option + R** and **Control + Shift + Space**. After pasting, you control when to submit the text in the destination app.

With **Wispr Flow** selected, Nuvra's shortcut opens the capture pill. Use Flow's own shortcut to dictate into it. Nuvra prepares and pastes the text after Flow's output settles.

If pasting fails, check the recording pill's status and **History** to recover the text. Nuvra also attempts to copy it to the clipboard.

## The app window

- **History** holds your latest 10 text entries, with controls to copy, delete, or clear them.
- **Models** manages speech models, the local translator, and your Codex or Claude Code selection.
- **Settings** controls writing mode, shortcuts, appearance, updates, and advanced speech options.

The interface supports English and Arabic. Change it under **Settings → Language**.

## Data and privacy

| Feature | Processing and storage |
| --- | --- |
| Local Whisper | Transcribes audio on your computer. |
| Cohere or Wispr Flow | Uses the selected provider's online speech service. |
| Local Arabic translation | Runs through Ollama when enabled and available. |
| Structured prompts and translation fallback | Sends text through your selected Codex or Claude Code provider using your signed-in account. |
| History | Stores recent text and destination information locally. |
| Settings and vocabulary | Stored on your computer. |

Local data locations:

| Platform | Settings, history, and vocabulary |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Arabic Relay` |
| macOS | `~/Library/Application Support/Nuvra` |

Windows retains the original data directory so existing installations keep their settings and history after the rename. Developers can override the data location on either platform with `ARABIC_RELAY_DATA_DIR`.

## Updates

Update preferences are under **Settings → Updates**. On Windows, automatic updates check GitHub Releases while Nuvra is running and verify the installer with the project's update-signing key before installation. When an update is ready, use **Restart and update**. You can also download the latest installer from the release page.

On macOS, updates notify you about new releases without downloading or installing them. Download the matching Mac asset, quit Nuvra, and replace the app in **Applications**. Choose **Off** to disable update checks.

## Run from source

On Windows, use Git and Python 3.12 or newer. The automated build uses Python 3.13.

```powershell
git clone https://github.com/Mahmoud6901/nuvra.git
cd nuvra
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m arabic_relay
```

For the NVIDIA GPU dependency, install the optional extra with `.\.venv\Scripts\python.exe -m pip install -e ".[gpu]"`.

The Python package remains `arabic_relay`. Both `nuvra` and `arabic-relay` are available as installed launch commands.

On macOS, use Python 3.13 from python.org or Homebrew. On Intel Macs, install the source-build prerequisites with `brew install openssl@3 rust` first. Clone the repository and run:

```sh
git clone https://github.com/Mahmoud6901/nuvra.git
cd nuvra
python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[build]'
.venv/bin/python -m arabic_relay
```

Development runs need microphone and Accessibility permissions for the Python host. The packaged app has its own permission identity.

### Tests

Install Node.js to run the interface tests. On Windows, prepare the small speech model used by the Python tests, then run the full suites from the repository root.

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-tiny', revision='d90ca5fe260221311c53c58e660288d3deb8d356')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Get-ChildItem tests\test_*.cjs | ForEach-Object {
    node $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Interface test failed in $($_.Name)" }
}
node tests/test_incremental_ui.cjs --quick
node tests/test_incremental_ui.cjs --quick --inline
node tests/test_voice.cjs --quick
```

On macOS, run the platform and shared delivery checks used by macOS CI:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_macos.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_paste.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_build_macos.py' -v
for test in tests/test_*.cjs; do node "$test" || exit 1; done
node tests/test_incremental_ui.cjs --quick
node tests/test_incremental_ui.cjs --quick --inline
node tests/test_voice.cjs --quick
```

Run the full Python suite on Windows because it includes Windows-specific integration tests. The release workflow includes the complete test and installer verification steps. See [the release workflow](.github/workflows/release.yml).

### Build the Windows installer

Install Inno Setup 6, then run the build from the repository root.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[build]"
.\scripts\build.ps1
```

The build creates `dist\Nuvra\Nuvra.exe` and `dist\Nuvra-Setup-<version>.exe` and runs the packaged app's self-test.

The release workflow also publishes an identical `ArabicRelay-Setup-<version>.exe` alias for older automatic updaters, with checksums and signatures for both filenames. New clients and the website prefer `Nuvra-Setup`; they also accept the older filename in existing releases. The installer keeps its original application ID and data locations so upgrades preserve settings, history and models. Run `python scripts/sign_release.py <tag>` to sign all installer aliases and publish the draft; mismatched aliases are rejected.

### Build macOS release assets

Run on the target architecture Mac with Python 3.13 and Apple's command-line tools:

```sh
OPENSSL_STATIC=1 .venv/bin/python -m pip install --force-reinstall --no-cache-dir 'cryptography>=43'
.venv/bin/python -m pip install -e '.[build]'
.venv/bin/python -m unittest discover -s tests -p 'test_macos.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_paste.py' -v
.venv/bin/python scripts/build_macos.py
```

Intel builds also require Homebrew's `openssl@3` and `rust` (`brew install openssl@3 rust`). Cryptography stopped publishing Intel macOS wheels in version 49, so pip builds it from source. `OPENSSL_STATIC=1` follows its [static build instructions](https://cryptography.io/en/latest/installation/#building-cryptography-on-macos) and prevents a collision with Python's bundled OpenSSL. Reinstalling without the cache replaces any previous dynamically linked build. Apple Silicon uses the upstream static wheel. The build script checks the installed extension's linkage before creating any build output.

The script creates `dist/Nuvra.app`, a DMG with an Applications shortcut, a ZIP, and SHA-256 checksums. It generates the macOS icon from the existing PNG, verifies the bundle's ad-hoc signature, and runs `--self-test` using a temporary data directory. That test checks the HTTP server and imports the packaged speech engine. A second `--self-test-gui` briefly opens a native window and checks that WebKit exposes the recording APIs. Neither test records audio or sends keys to another app. Build in a logged-in graphical macOS session.

The macOS workflow builds both architectures on branch pushes, pull requests and manual runs. Tagged releases wait for these builds and attach their assets to the existing draft release. The maintainer's offline Windows signing/publishing step remains unchanged. Apple Developer ID signing and notarization require a separate maintainer setup. The Tests workflow also supports branch pushes and manual runs, so contributors can check their branch in their own fork while an upstream workflow awaits approval.

Before publishing a macOS build, test microphone permission, dictation, Return/Escape, clipboard restoration, and pasting into a real text field on each architecture. Automated tests mock global input and use a separate test clipboard.

## Report a problem

Include the app version, steps to reproduce the problem, and what you expected. **Settings → Advanced → Copy diagnostics** provides environment details to help investigate. Review your report for personal information before posting it. [Open a bug report](https://github.com/Mahmoud6901/nuvra/issues/new?template=bug_report.yml).

## License

Nuvra is released under the [MIT License](LICENSE).
