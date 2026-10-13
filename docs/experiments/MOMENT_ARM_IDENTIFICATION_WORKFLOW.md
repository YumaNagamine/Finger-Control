# Moment arm identification workflow

## Purpose

This document defines the adopted procedure for identifying tendon moment arms in a robot-finger sample.

The final method fits tendon excursion as a fourth-order polynomial of joint flexion angle and differentiates that polynomial analytically. Comparison algorithms based on time derivatives are not part of this workflow.

## Adopted method

For each target joint and tendon, fit

\[
L(\theta)=a_4\theta^4+a_3\theta^3+a_2\theta^2+a_1\theta+a_0,
\]

where \(\theta\) is the joint flexion angle in radians and \(L\) is tendon excursion in millimetres.

Calculate the moment arm from

\[
r(\theta)=\frac{dL}{d\theta}
=4a_4\theta^3+3a_3\theta^2+2a_2\theta+a_1.
\]

The moment-arm unit is `mm/rad`.

## Do not use for final results

- Do not calculate moment arm from `(dL/dt) / (dtheta/dt)`.
- Do not run fitting-method comparisons for final data production.
- Do not split the recording into repetitions.
- Do not replace samples with angle-bin medians.
- Do not average the signal in the time direction before fitting.
- Do not average coefficients from different recordings unless a later protocol explicitly requires it.

The following scripts are comparison or legacy tools and are not part of this workflow:

- `src/dataProcessing/analyze_single_recording_moment_arm.py`
- `src/dataProcessing/compare_dip_fitting_methods.py`
- `src/dataProcessing/calculate_moment_arm_from_differences.py`

## Camera roles and tendon mapping

- Camera 0: joint-angle measurement with DeepLabCut
- Camera 1: tendon-excursion measurement using ArUco markers
- `pair_timestamps.csv`: correspondence between frames from the two cameras

| ArUco ID | Processed CSV column | Tendon |
| ---: | --- | --- |
| 0 | `weight_0_disp_mm` | FDP |
| 1 | `weight_1_disp_mm` | FDS |
| 2 | `weight_2_disp_mm` | ED |
| 3 | `weight_3_disp_mm` | PI |
| 4 | `weight_4_disp_mm` | DI |
| 5 | `weight_5_disp_mm` | LUM |

## Environment

Run commands in `cmd.exe` from the repository root.

```bat
cd /d C:\Users\ynaga\Desktop\Finger-Control
call C:\Users\ynaga\anaconda3\condabin\conda.bat activate FingerControl
```

The expected interpreter is:

```text
C:\Users\ynaga\anaconda3\envs\FingerControl\python.exe
```

## Step 1: record a target-joint experiment

Use the optimized dual-camera recorder:

```bat
python src\observation\vision\record_dual_cameras.py
```

Do not use `record_dual_cameras_legacy.py` for new recordings.

The recorder normally creates:

```text
logs\dual_camera\recordings\dual_record_YYYYMMDD_HHMMSS\
  cam0.avi
  cam1.avi
  cam0_timestamps.csv
  cam1_timestamps.csv
  pair_timestamps.csv
  session_manifest.json
```

Record MCP, PIP and DIP experiments separately. Keep a clear correspondence between the recording directory and target joint.

After recording, verify `session_manifest.json`, especially:

- camera resolution
- saved frame counts
- actual frame rates
- dropped-frame counts

## Step 2: configure dual-camera processing

Create or update a processing configuration for each target-joint recording. The relevant script is:

```text
src/observation/vision/process_dual_recording.py
```

Existing examples are:

- MCP: `src/observation/vision/config_dual_camera_process_20260912_193519.json`
- PIP: `src/observation/vision/config_dual_camera_process_pip.json`
- DIP: `src/observation/vision/config_dual_camera_process_dip.json`

The configuration must reference the actual `cam0`, `cam1` and `pair_timestamps.csv` files. Give `output.filename_prefix` a name containing exactly one of `MCP`, `PIP` or `DIP`, because the fitting script infers the target joint from the processed CSV filename.

## Step 3: extract joint angles and tendon excursion

Run:

```bat
python src\observation\vision\process_dual_recording.py --config <processing-config.json>
```

This performs:

1. DeepLabCut inference on Camera 0.
2. DIP, PIP and MCP angle calculation.
3. ArUco displacement measurement on Camera 1.
4. Camera-frame synchronization using `pair_timestamps.csv`.
5. Processed CSV and optional overlay-video output.

### Fixed DLC reference keypoint

The offline dual-camera processor uses the fixed pixel coordinate `(1104, 728)` in the 1600 x 1200 Camera 0 image for the `meta1` reference point. It does not use the DLC-predicted `meta1` position when calculating the joint angles. This behavior affects the branch that introduced the fixed-coordinate feature and every branch that incorporates that change; confirm that the active processing branch contains the intended implementation before comparing results.

The fixed coordinate is valid only when the experimental setup matches the setup used on 2026-09-15 (`20260915`). Before processing, verify the camera placement, camera roles, image resolution, field of view, calibration/undistortion settings and the position of the hand or finger in the Camera 0 image. If the setup differs, revalidate or update the fixed coordinate before accepting the joint-angle results.

Before fitting, inspect the overlay video and confirm that joint keypoints and ArUco markers are tracked correctly.

## Step 4: convert to flexion angle

The fitting script calculates

\[
\theta_{flexion}=180^\circ-\theta_{measured}
\]

and then

\[
\theta_{rad}=\theta_{flexion}\frac{\pi}{180}.
\]

It adds flexion-angle columns for DIP, PIP and MCP and renames the six displacement columns to the tendon names.

## Step 5: apply the fitting range

Only samples inside the target joint's range are used:

| Joint | Fitting range |
| --- | ---: |
| MCP | -10 to 90 deg |
| PIP | 0 to 90 deg |
| DIP | -10 to 40 deg |

All finite samples inside the range are fitted together as one dataset.

## Step 6: fit excursion and calculate moment arm

Run:

```bat
python src\dataProcessing\postprocess_dual_processed_csv.py ^
  --csv <dual-processed-csv> ^
  --output-root <sample-specific-output-directory>
```

Use a sample-specific output root so that data from different robot fingers are not mixed.

The script:

1. validates the input columns;
2. infers the target joint from the filename;
3. calculates flexion angles in degrees and radians;
4. keeps data inside the target-joint range;
5. fits a fourth-order polynomial for each tendon using all valid points;
6. differentiates the polynomial analytically;
7. saves plots, processed data and coefficients.

## Outputs

Each processed recording produces a directory containing:

- `processed_csv.csv`
- `joint_angles_vs_time.png`
- `mcp_flexion_vs_excursion_fit.png`, `pip_flexion_vs_excursion_fit.png` or `dip_flexion_vs_excursion_fit.png`
- `mcp_moment_arm.png`, `pip_moment_arm.png` or `dip_moment_arm.png`
- `polynomial_coefficients.csv`

The excursion plot reports the number of valid samples, R-squared and RMSE for every tendon. The coefficient CSV contains both the fourth-order excursion coefficients and the differentiated third-order moment-arm coefficients.

## Final checks

- The input filename identifies exactly one joint.
- The active processing branch contains the fixed-`meta1` behavior, if that behavior is intended for the dataset.
- The experimental setup matches the 2026-09-15 (`20260915`) setup, or the fixed `meta1` coordinate has been revalidated for the changed setup.
- The expected flexion range is covered adequately.
- The fitted curve follows the measured excursion points without obvious extrapolation artefacts.
- The moment-arm curve is evaluated only over the adopted joint range.
- Units are degrees for displayed angle, radians for polynomial evaluation, millimetres for excursion and `mm/rad` for moment arm.
- Final results came from the fourth-order excursion fit, not a time-derivative comparison script.
