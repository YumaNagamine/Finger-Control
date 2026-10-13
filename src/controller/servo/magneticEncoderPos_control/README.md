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
`MULTITURN_UNCERTAIN` status.
