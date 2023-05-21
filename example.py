import platform

is_windows = platform.platform().lower().startswith("windows")
print(is_windows)
