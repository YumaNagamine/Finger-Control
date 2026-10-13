# Tendon excursion-joint displacement workflow

## Purpose

This document defines the procedure for measuring how MCP, PIP and DIP joint positions change when one robot-finger tendon is moved through a prescribed excursion.

This is a tendon-specific experiment. Record and process FDP, FDS, ED and any additional tendons separately.

## Required processing order

For new 800 x 600 recordings, use this order:

1. Record with the optimized dual-camera recorder.
2. Correct lens distortion at 800 x 600.
3. Upscale the corrected video to 1600 x 1200.
4. Extract joint angles and tendon excursion with the dual-camera processor.
5. Clean the selected tendon signal and calculate joint displacement.
6. Plot tendon excursion against MCP, PIP and DIP displacement.

Do not run the dual-camera analysis directly on the uncorrected 800 x 600 videos.

## Camera roles and tendon mapping

- Camera 0: DeepLabCut joint-angle view
- Camera 1: ArUco tendon-excursion view
- `pair_timestamps.csv`: synchronized frame mapping

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

## Step 1: record one tendon

Use:

```bat
python src\observation\vision\record_dual_cameras.py
```

Do not use:

```text
src/observation/vision/record_dual_cameras_legacy.py
```

The active recording configuration is:

```text
src/observation/vision/config_dual_camera_record.json
```

With the `middle` resolution profile, the camera parameter resolver selects the 800 x 600 configurations under:

```text
src/observation/camera/params/
```

Record one tendon at a time. Keep the other tendons in their defined initial condition. Use a directory name that identifies the tendon, for example:

```text
logs/dual_camera/recordings/FDP_800x600
logs/dual_camera/recordings/FDS_800x600
logs/dual_camera/recordings/ED_800x600
```

After recording, check `session_manifest.json` for frame counts, actual frame rates and dropped frames.

## Step 2: undistort and upscale

Use the combined one-pass script:

```text
src/observation/vision/upscale_and_undistort_video.py
```

For Camera 0, use the 800 x 600 DLC-camera calibration:

```text
src/observation/camera/params/camera_calibration_dlc.json
```

For Camera 1, use the 800 x 600 weight-camera calibration:

```text
src/observation/camera/params/camera_calibration_weight.json
```

Example for FDP:

```bat
mkdir logs\dual_camera\recordings\FDP_upscaled_1600x1200

python src\observation\vision\upscale_and_undistort_video.py ^
  logs\dual_camera\recordings\FDP_800x600\cam0.avi ^
  logs\dual_camera\recordings\FDP_upscaled_1600x1200\cam0.mp4 ^
  src\observation\camera\params\camera_calibration_dlc.json

python src\observation\vision\upscale_and_undistort_video.py ^
  logs\dual_camera\recordings\FDP_800x600\cam1.avi ^
  logs\dual_camera\recordings\FDP_upscaled_1600x1200\cam1.mp4 ^
  src\observation\camera\params\camera_calibration_weight.json
```

The script first applies distortion correction at the calibrated 800 x 600 resolution, then upscales the corrected frames to 1600 x 1200 using Lanczos interpolation. Do not upscale first and then apply the 800 x 600 calibration.

The output FPS and frame count must match the corresponding input video.

### Alternative two-step tools

The repository also contains:

- `src/observation/vision/undistort_dual_recording.py`
- `src/observation/vision/upscale_video.py`

`undistort_dual_recording.py` replaces `cam0.avi` and `cam1.avi` with corrected versions after validation and renames the originals to `cam0_distorted.avi` and `cam1_distorted.avi`. Prefer the combined one-pass script when the original recording should remain untouched.

## Step 3: retain synchronization CSV files

Upscaling does not change frame order, FPS or timestamps. Copy the synchronization files into the upscaled recording directory:

```bat
copy logs\dual_camera\recordings\FDP_800x600\pair_timestamps.csv ^
     logs\dual_camera\recordings\FDP_upscaled_1600x1200\pair_timestamps.csv

copy logs\dual_camera\recordings\FDP_800x600\cam0_timestamps.csv ^
     logs\dual_camera\recordings\FDP_upscaled_1600x1200\cam0_timestamps.csv

copy logs\dual_camera\recordings\FDP_800x600\cam1_timestamps.csv ^
     logs\dual_camera\recordings\FDP_upscaled_1600x1200\cam1_timestamps.csv
```

Repeat the same procedure for FDS, ED and any other tendon.

## Step 4: configure dual-camera processing

The processor is:

```text
src/observation/vision/process_dual_recording.py
```

Existing tendon-specific configuration examples are:

- `src/observation/vision/config_dual_camera_process_fdp_upscaled.json`
- `src/observation/vision/config_dual_camera_process_fds_upscaled.json`
- `src/observation/vision/config_dual_camera_process_ed_upscaled.json`

For every new robot-finger sample, verify that each configuration points to the new sample's:

- upscaled `cam0.mp4`
- upscaled `cam1.mp4`
- copied `pair_timestamps.csv`
- sample-specific output directories

Do not rely on an old recording path merely because the tendon name matches.

## Step 5: extract angles and excursion

Run once per tendon:

```bat
python src\observation\vision\process_dual_recording.py ^
  --config src\observation\vision\config_dual_camera_process_fdp_upscaled.json
```

Replace the configuration filename for FDS and ED.

The processor performs:

1. DeepLabCut inference on the upscaled Camera 0 video.
2. DIP, PIP and MCP angle calculation.
3. ArUco displacement measurement on the upscaled Camera 1 video.
4. Frame synchronization using `pair_timestamps.csv`.
5. Processed CSV and overlay-video output.

### Fixed DLC reference keypoint

The offline dual-camera processor uses the fixed pixel coordinate `(1104, 728)` in the 1600 x 1200 Camera 0 image for the `meta1` reference point. It does not use the DLC-predicted `meta1` position when calculating the joint angles. This behavior affects the branch that introduced the fixed-coordinate feature and every branch that incorporates that change; confirm that the active processing branch contains the intended implementation before comparing results.

The fixed coordinate is valid only when the experimental setup matches the setup used on 2026-09-15 (`20260915`). Before processing, verify the camera placement, camera roles, image resolution, field of view, calibration/undistortion settings and the position of the hand or finger in the Camera 0 image. If the setup differs, revalidate or update the fixed coordinate before accepting the joint-angle results.

Inspect the overlay video before accepting the CSV. Confirm that the DLC keypoints and the selected tendon marker are tracked throughout the active movement.

## Step 6: calculate joint displacement and plot

Use:

```text
src/dataProcessing/plot_joint_flexion_vs_excursion.py
```

Example:

```bat
python src\dataProcessing\plot_joint_flexion_vs_excursion.py ^
  --csv <dual-processed-FDP.csv> ^
  --tendon FDP
```

Run corresponding commands with `--tendon FDS` and `--tendon ED`.

The script performs the following processing:

1. Selects the ArUco displacement column associated with `--tendon`.
2. Converts every measured joint angle to flexion angle using `180 deg - measured angle`.
3. Calculates the baseline flexion angle from the first five frames by default.
4. Interpolates internal gaps of at most five frames.
5. Applies a centred 61-frame moving average by default.
6. Detects the active-motion interval between 2% and 98% of the smoothed excursion range.
7. Calculates MCP, PIP and DIP displacement relative to the baseline.
8. Plots joint displacement in degrees against the selected tendon excursion in millimetres.

The main adjustable command-line parameters are:

- `--window`: positive odd moving-average window; default `61`
- `--baseline-frames`: number of initial frames used for the baseline; default `5`
- `--rows`: optional input-row limit; `0` means all rows
- `--no-raw-overlay`: omit the faint raw-data traces from the plot

## Outputs

The current plotting script writes:

- cleaned CSV under `logs/dual_camera/processed_csv/cleaned`
- PNG under `logs/dual_camera/processed_plot`

The filenames contain the tendon name, smoothing-window size and processing timestamp.

Because these output roots are currently defined inside the script, move or copy accepted final results into a sample-specific results directory after verification, or update the script separately if configurable output roots are required.

## Final checks

- The recording was created with `record_dual_cameras.py`, not the legacy recorder.
- Both videos were corrected with their matching 800 x 600 calibration before upscaling.
- Both upscaled videos are 1600 x 1200 and retain the original FPS and frame count.
- `pair_timestamps.csv` belongs to the same recording session as the two videos.
- The processing configuration points to the correct sample and tendon.
- The `--tendon` argument matches the tendon that was physically moved.
- The selected ArUco marker is visible during the active interval.
- The active processing branch contains the fixed-`meta1` behavior, if that behavior is intended for the dataset.
- The experimental setup matches the 2026-09-15 (`20260915`) setup, or the fixed `meta1` coordinate has been revalidated for the changed setup.
- The baseline frames represent the intended initial posture.
- The processed plot shows excursion in millimetres and joint displacement in degrees.
