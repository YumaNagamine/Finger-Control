# Magnetic-encoder position PID control

This package controls one STS3215 servo with a PID whose measurement is the
AS5048A bridge's continuous `relative_multi_turn_deg` value. The servo remains
the actuator: the PID output is sent as a servo speed command.

Run a dry-run plan from the repository root after activating the
`FingerControl` environment:

```text
cmd.exe /d /s /c "call C:\Users\ynaga\anaconda3\condabin\conda.bat activate FingerControl && python -m controller.servo.magneticEncoderPos_control --servo-port COM3 --encoder-port COM5 --servo-id 4 --target-position-deg 90 --kp 1 --ki 0 --kd 0 --max-speed 100 --hard-position-error-deg 180"
```

Add `--execute` only after checking the sign, target, mechanical limits, and
emergency stop procedure with the tendon unloaded. The controller rejects
single-turn-only samples and stops on stale servo/encoder feedback or a
`MULTITURN_UNCERTAIN` status in direct-target mode.

To return one servo to the position saved by the Force Control GUI, pass
`--return-to-calibration-position` instead of `--target-position-deg`. The
saved record supplies the encoder channel and directions. For example, from
the repository root:

```text
cmd.exe /d /s /c "call C:\Users\ynaga\anaconda3\condabin\conda.bat activate FingerControl && python -m controller.servo.magneticEncoderPos_control --servo-port COM3 --encoder-port COM5 --servo-id 4 --return-to-calibration-position --kp 1 --max-speed 100 --hard-position-error-deg 720 --execute"
```

Remembered-position return accepts `MULTITURN_UNCERTAIN` only when the
continuous multi-turn angle is present and finite. Other encoder statuses,
missing angles, and position errors beyond the configured hard limit still stop
the controller.

The CLI stops when position stays within the configured `--target-tolerance-deg`
for `--settle-time-s`, or stops with an error after `--return-timeout-s`. Use
`--calibration-position-file` when the GUI saved positions outside the default
calibration directory. Existing records without direction metadata use the
servo's calibration CSV. `--encoder-direction` can override the saved encoder
direction while preserving the saved target in the new direction.
`--control-sign` can override the saved control direction.
