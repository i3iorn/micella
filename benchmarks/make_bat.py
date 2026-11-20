content = (
    "@echo off\r\n"
    'call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community'
    '\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1\r\n'
    "C:\\Users\\micro\\.venv-littlebit\\Scripts\\python.exe %*\r\n"
)
with open(r"C:\Users\micro\Code\Projects\micella\benchmarks\run_msvc.bat", "w") as fh:
    fh.write(content)
print("written")
