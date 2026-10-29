# Control and RL architecture

## 確認した事実

下位制御はAbsolute Excursion目標を磁気Encoder角度へ変換し、PIDが速度指令を生成する。FFとFF+FBもSession原点基準のAbsolute目標を生成する。この下位構造は変更しない。

## 採用するAction契約

Direct RLのActionは6次元Incremental Excursionである。

```text
requested_delta_mm
  -> Increment上限
  -> requested_target_mm = previous_pid_target_mm + requested_delta_mm
  -> Absolute Excursion上下限
  -> 張力・関節角度・通信・鮮度検査
  -> filtered_target_mm
  -> PIDへ反映
  -> pid_reflected_delta_mm
```

Critic、Replay Buffer、BC教師ラベルには`pid_reflected_delta_mm`を使う。これはソフトウェア上でPID目標へ反映された値であり、物理サーボ適用確認ではない。

`IncrementalExcursionController`はDirectとResidualの共通境界を提供する。Directでは前回PID目標へPolicy増分を加える。Residualでは将来、Base ControllerのAbsolute目標へ補正を加えられる。Residual学習自体は今回実装しない。

## Observation契約

基本Observationは次の24次元で固定する。

```text
0:3    current_joint_rad
3:6    goal_joint_rad - current_joint_rad
6:12   measured_excursion_mm
12:18  tendon_tension_n
18:24  previous_pid_reflected_delta_mm
```

Goalそのもの、初期姿勢、Sensor age、元Sample ID、Calibration hashはMetadataへ保存する。必要性が実測で確認された場合のみ履歴を追加する。

## Fail-safe

- Action欠損、次元不一致、NaN、Infinityは前回PID目標を保持して拒否する。
- 通信状態がFalseまたは不明なら拒否する。
- Safety Filterが状態異常を返した場合は前回PID目標を保持する。
- 安全な保持が物理的に成立するとは仮定しない。実機経路では既存のラッチ停止判断を維持する。
- PC書込Receiptから物理停止・張力安全を推測しない。

## Mock Environment

`GoalConditionedMockEnv`は`reset`、`step`、Observation、Reward、境界、安全統合、Event sinkを提供する。内部DynamicsはSoftware統合テスト専用で、Moment Arm実機物理モデルでも性能証拠でもない。

