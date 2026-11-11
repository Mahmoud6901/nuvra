; Per-user installer: ISCC /DAppVersion=<version> installer\ArabicRelay.iss (run through scripts\build.ps1).
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{B981CB91-79B0-4049-ABF8-C2A11A94560B}
AppName=Nuvra
AppVersion={#AppVersion}
AppVerName=Nuvra {#AppVersion}
AppPublisher=Mahmoud Adel Hassan
AppPublisherURL=https://github.com/Mahmoud6901/nuvra
DefaultDirName={autopf}\Nuvra
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\Nuvra.exe
SetupIconFile=..\src\arabic_relay\assets\relay.ico
OutputDir=..\dist
; Existing clients find this exact asset name when checking for updates.
OutputBaseFilename=ArabicRelay-Setup-{#AppVersion}
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\Nuvra\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Upgrade direct executable shortcuts as well as the new Nuvra shortcut.
Source: "..\dist\Nuvra\Nuvra.exe"; DestDir: "{app}"; DestName: "Arabic Relay.exe"; Flags: ignoreversion; Check: HasLegacyExecutable
Source: "..\build\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: NeedsWebView2

[InstallDelete]
Type: files; Name: "{autoprograms}\Arabic Relay.lnk"
Type: files; Name: "{autodesktop}\Arabic Relay.lnk"

[Icons]
Name: "{autoprograms}\Nuvra"; Filename: "{app}\Nuvra.exe"
Name: "{autodesktop}\Nuvra"; Filename: "{app}\Nuvra.exe"; Check: CreateDesktopShortcut

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Installing Microsoft Edge WebView2..."; Flags: waituntilterminated; Check: NeedsWebView2
Filename: "{app}\Nuvra.exe"; Description: "{cm:LaunchProgram,Nuvra}"; Flags: nowait postinstall skipifsilent
; Restart and update passes /RELAUNCH=1 so a silent update reopens the app.
Filename: "{app}\Nuvra.exe"; Flags: nowait; Check: Relaunch

[UninstallDelete]
; Only our own folder: the user may have picked an existing folder to install into.
Type: filesandordirs; Name: "{app}\_internal"
Type: dirifempty; Name: "{app}"

[Code]
const
  WebView2Client = 'Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

var
  LegacyDesktopShortcut: Boolean;

function InitializeSetup: Boolean;
begin
  LegacyDesktopShortcut := FileExists(ExpandConstant('{autodesktop}\Arabic Relay.lnk'));
  Result := True;
end;

function CreateDesktopShortcut: Boolean;
begin
  Result := WizardIsTaskSelected('desktopicon') or LegacyDesktopShortcut;
end;

function HasLegacyExecutable: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\Arabic Relay.exe'));
end;

function HasWebView2(Root: Integer; Key: String): Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(Root, Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0');
end;

function NeedsWebView2: Boolean;
begin
  Result := not (HasWebView2(HKLM, 'SOFTWARE\WOW6432Node\' + WebView2Client) or HasWebView2(HKCU, 'Software\' + WebView2Client));
end;

function Relaunch: Boolean;
begin
  Result := ExpandConstant('{param:RELAUNCH|0}') = '1';
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  { The user's vocabulary and models stay unless they say Yes; silent uninstalls never delete them. }
  if (CurUninstallStep = usPostUninstall) and not UninstallSilent then
    if MsgBox('Also delete your vocabulary and downloaded models?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(ExpandConstant('{localappdata}\Arabic Relay'), True, True, True);
end;
