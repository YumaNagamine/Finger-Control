# Experiment and dataset specification

## 実験記録

Raw Eventを正本とし、CSV、評価JSON、NPZは派生Artifactとする。最低限、次を同じExperiment IDで結合する。

- Controller type、D0/D1/D2、Goal ID、初期姿勢、目標姿勢
- Target posture trajectory、measured joint angles
- FF target、FB correction、exploration correction
- requested delta、requested target、filtered target
- PID-reflected targetとdelta
- measured excursion、tendon tension
- Decision、Sensor capture、Inference、PID反映、Transportの各時刻
- Sensor freshness、Safety reasons、Stop reason
- Transport status、physical applied status

`physical_applied_status`は観測できない限り`unknown`または`unobserved`とする。

## P2P評価

`p2p_evaluation.py`は以下を計算する。

- 各関節の最終誤差
- 3関節誤差のL2統合値
- 設定可能な許容誤差と保持時間による成功判定
- 整定時間
- 保持区間の関節RMSE
- 目標軌道追従RMSE
- 最大張力
- Safety violation数
- FB補正RMSと最大絶対量

Timestamp不整合、Shape不一致、非有限値は`evaluation_quality=invalid`とする。補助系列だけが不正な場合は`degraded`とし、理由を残す。

## Transition Schema 3

```text
(o_t, a_t, r_t, o_t+1, terminated, truncated)
```

- `o_t`: 24次元Observation
- `a_t`: `pid_reflected_delta_mm`
- `r_t`: 再計算可能な姿勢誤差Reward
- `terminated`: 明示されたGoal達成、または定義済みMDP終端
- `truncated`: 時間上限、Operator停止、通信・Sensor中断

MetadataにはEpisode ID、Goal ID、D0/D1/D2、要求値、Filter後値、PID反映値、時刻、鮮度、Clipping、終了理由、初期化条件を保持する。

旧Absolute Sessionは、現在目標と前回目標の差からIncrementを再構成する。ただしAction分解と物理適用確認がないため、`legacy_reconstructed`品質として本学習データと区別することを推奨する。

RewardはRaw姿勢とGoalからOffline再計算できる。Action期間に対応するCaptureがない場合は補完せずTransitionを棄却する。

