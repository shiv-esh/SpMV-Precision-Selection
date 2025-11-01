@echo off
set "program_path=C:\Users\lenovo\Desktop\MpSpMV\src\Mp_SpMV_c.exe"
set "files_dir=C:\Users\lenovo\Desktop\Project\mtx_files"

for %%F in ("%files_dir%\*.*") do (
    "%program_path%" "%%F"
)

echo All files processed.
