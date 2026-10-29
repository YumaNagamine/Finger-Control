# Torque zero calibration

Torque zero calibration is part of the Force Control GUI. Use its Speed control
tab to place a tendon at the repeatable loose position, stop every servo, then
use the Torque calibration tab.

`Calibrate (set current force as zero)` averages the current angle difference
for one second and saves `b = -k * mean(delta theta)`. The fitted slope `k` in
the raw spring-torque CSV is never changed.

`Remember encoder position for calibration` stores the current AS5048A
multi-turn position, encoder direction, and the Force Control tab's current
control sign. Use `Return to
remembered position` in the same row to move that servo back to the stored
position. Return is available while force control is stopped and all manual
speed commands are zero. It uses a position PID limited to speed 100, stops
within 1 degree after a 0.25 s settle period, and fails if the target is 720
degrees or more away or is not reached within 60 seconds. It can use a finite
continuous multi-turn angle with `MULTITURN_UNCERTAIN` status; other encoder
statuses still stop the return. `STOP ALL` cancels an active return.

The same saved position can be used from the magnetic-encoder position CLI with
`--return-to-calibration-position`. Keep the tendon unloaded, check the saved
control direction and movement path, and keep the emergency stop accessible
before starting either return action.
