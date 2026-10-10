"""Windows global push-to-talk bridge for the local voice dashboard.

The helper installs low-level keyboard and mouse hooks, then forwards only the
configured PTT button's pressed/released state to the loopback dashboard. It
does not capture text, inject input, or communicate outside localhost.
"""

import argparse
import ctypes
import json
import queue
import sys
import threading
import time
import urllib.request
from ctypes import wintypes


WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


def keyboard_code(vk_code):
    names = {
        0xA0: "ShiftLeft",
        0xA1: "ShiftRight",
        0xA2: "ControlLeft",
        0xA3: "ControlRight",
        0xA4: "AltLeft",
        0xA5: "AltRight",
        0x20: "Space",
        0x0D: "Enter",
        0x09: "Tab",
        0x1B: "Escape",
    }
    if vk_code in names:
        return names[vk_code]
    if 0x41 <= vk_code <= 0x5A:
        return "Key" + chr(vk_code)
    if 0x30 <= vk_code <= 0x39:
        return "Digit" + chr(vk_code)
    if 0x60 <= vk_code <= 0x69:
        return "Numpad" + str(vk_code - 0x60)
    return ""


def post_event(base_url, held):
    body = json.dumps({"held": bool(held), "source": "global-helper"}).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/api/ptt-event",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        response.read()


def read_binding(base_url):
    with urllib.request.urlopen(base_url + "/api/ptt-state", timeout=2) as response:
        return json.loads(response.read().decode("utf-8")).get("binding") or "ShiftLeft"


def main():
    if sys.platform != "win32":
        raise SystemExit("The global PTT helper currently requires Windows.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", required=True, help="Dashboard address, for example http://127.0.0.1:8790.")
    args = parser.parse_args()

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    binding = {"value": "ShiftLeft"}
    event_queue = queue.Queue()
    pressed = {"value": False}

    def refresh_binding():
        while True:
            try:
                binding["value"] = read_binding(args.dashboard)
            except Exception:
                pass
            time.sleep(1)

    def send_events():
        while True:
            held = event_queue.get()
            try:
                post_event(args.dashboard, held)
            except Exception:
                pass

    threading.Thread(target=refresh_binding, daemon=True).start()
    threading.Thread(target=send_events, daemon=True).start()

    result_type = ctypes.c_ssize_t
    keyboard_proc_type = ctypes.WINFUNCTYPE(
        result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
    )
    mouse_proc_type = ctypes.WINFUNCTYPE(
        result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
    )
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = ctypes.c_void_p
    user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = result_type
    user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]

    def change_state(code, is_down):
        if not code or code != binding["value"]:
            return
        if pressed["value"] == is_down:
            return
        pressed["value"] = is_down
        event_queue.put(is_down)

    @keyboard_proc_type
    def keyboard_proc(n_code, w_param, l_param):
        if n_code >= 0:
            data = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                change_state(keyboard_code(data.vkCode), True)
            elif w_param in (WM_KEYUP, WM_SYSKEYUP):
                change_state(keyboard_code(data.vkCode), False)
        return user32.CallNextHookEx(None, n_code, w_param, l_param)

    @mouse_proc_type
    def mouse_proc(n_code, w_param, l_param):
        if n_code >= 0:
            data = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            mapping = {
                WM_LBUTTONDOWN: ("Mouse0", True),
                WM_LBUTTONUP: ("Mouse0", False),
                WM_MBUTTONDOWN: ("Mouse1", True),
                WM_MBUTTONUP: ("Mouse1", False),
                WM_RBUTTONDOWN: ("Mouse2", True),
                WM_RBUTTONUP: ("Mouse2", False),
            }
            event = mapping.get(w_param)
            if w_param in (WM_XBUTTONDOWN, WM_XBUTTONUP):
                button = (data.mouseData >> 16) & 0xFFFF
                event = ("Mouse3" if button == 1 else "Mouse4", w_param == WM_XBUTTONDOWN)
            if event:
                change_state(*event)
        return user32.CallNextHookEx(None, n_code, w_param, l_param)

    module = kernel32.GetModuleHandleW(None)
    keyboard_hook = user32.SetWindowsHookExW(
        WH_KEYBOARD_LL, ctypes.cast(keyboard_proc, ctypes.c_void_p), module, 0
    )
    mouse_hook = user32.SetWindowsHookExW(
        WH_MOUSE_LL, ctypes.cast(mouse_proc, ctypes.c_void_p), module, 0
    )
    if not keyboard_hook or not mouse_hook:
        raise ctypes.WinError()

    print("Global PTT helper ready.", flush=True)
    message = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
    finally:
        if pressed["value"]:
            try:
                post_event(args.dashboard, False)
            except Exception:
                pass
        user32.UnhookWindowsHookEx(keyboard_hook)
        user32.UnhookWindowsHookEx(mouse_hook)


if __name__ == "__main__":
    main()
