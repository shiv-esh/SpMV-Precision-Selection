@echo off
set "source_dir=C:\Users\lenovo\Desktop\Project\matrices\extract"
set "destination_dir=C:\Users\lenovo\Desktop\Project\mtx_files"

for /D %%i in ("%source_dir%\*") do (
    for %%j in ("%%i\*") do (
        move "%%j" "%destination_dir%"
    )
)

echo Files moved successfully.
