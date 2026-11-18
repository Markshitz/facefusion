; Inno Setup 6 script. Build: iscc /DFlavor=directml /DVersion=0.1.0 packaging\installer.iss
#ifndef Flavor
  #define Flavor "directml"
#endif
#ifndef Version
  #define Version "0.1.0"
#endif

[Setup]
AppId={{6F0C9C59-3E8B-4C5D-9E0A-5A1B2C3D4E5F}
AppName=FaceFusion Live
AppVersion={#Version}
AppPublisher=FaceFusion Live contributors
DefaultDirName={localappdata}\Programs\FaceFusion Live
DefaultGroupName=FaceFusion Live
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=FaceFusionLive-{#Version}-{#Flavor}-setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
LicenseFile=..\docs\RESPONSIBLE_USE.txt
UninstallDisplayName=FaceFusion Live

[Files]
Source: "..\dist\FaceFusionLive\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\FaceFusion Live"; Filename: "{app}\FaceFusionLive.exe"
Name: "{group}\Uninstall FaceFusion Live"; Filename: "{uninstallexe}"
Name: "{userdesktop}\FaceFusion Live"; Filename: "{app}\FaceFusionLive.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Run]
Filename: "{app}\FaceFusionLive.exe"; Description: "Launch FaceFusion Live"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var dataDir: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    dataDir := ExpandConstant('{localappdata}\FaceFusionLive');
    if DirExists(dataDir) then
      if (not UninstallSilent) and (MsgBox('Also delete downloaded models, settings and logs?' + #13#10 + dataDir,
                                    mbConfirmation, MB_YESNO) = IDYES) then
        DelTree(dataDir, True, True, True);
  end;
end;
