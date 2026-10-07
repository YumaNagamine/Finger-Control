# Torque intercept calibration GUI

This GUI recalibrates only the torque-intercept `b` while keeping the existing
per-servo slope `k` fixed.

Start it from the project root after activating the `FingerControl` Conda
environment:

```cmd
set PYTHONPATH=src
python -m controller.servo.torque_calibration
```

The GUI does not send movement commands. It reads synchronized Servo and
AS5048A telemetry, while the force-gauge value is entered manually in newtons.
For each selected Servo ID:

1. Hold the measured force steadily.
2. Enter the force-gauge reading and torque direction.
3. Click `Capture point`.
4. Repeat for several force points and click `Calculate b`.

After all six Servo IDs have a calculated value, click
`Update calibrated intercepts for all six servos`.

The result is saved to
`third_party/ServoHardwareAPI/calibration/current/torque_intercept_calibration.json`.
The raw spring-torque CSV files, their slopes, and
`as5048a_unloaded_zero_offsets.json` are not changed. Existing force-control
paths prefer this file, then fall back to the unloaded-zero correction, then
to the intercept fitted from the original CSV points.
