# Torque zero calibration

Torque zero calibration is part of the Force Control GUI. Use its Speed control
tab to place a tendon at the repeatable loose position, stop every servo, then
use the Torque calibration tab.

`Calibrate (set current force as zero)` averages the current angle difference
for one second and saves `b = -k * mean(delta theta)`. The fitted slope `k` in
the raw spring-torque CSV is never changed.

`Remember encoder position for calibration` stores the current AS5048A
multi-turn position for future automatic return-to-calibration support. It does
not move the servo automatically.
