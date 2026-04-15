from __future__ import annotations

import tkinter as tk

from servo.servoGUI_api.app import ServoControlGUIAPI


def main() -> None:
    root = tk.Tk()
    ServoControlGUIAPI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
