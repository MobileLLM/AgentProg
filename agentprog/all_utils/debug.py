import os

need_breakpoint = False
if os.environ.get("DEBUG", "0") == "1":
    need_breakpoint = True
need_runtime_point = False
