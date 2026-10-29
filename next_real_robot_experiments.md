# Next real-robot experiments

本作業では実機を自動駆動していない。以下は実機実験前の手動Review項目と、次に収集すべきデータである。

## 実験前の安全確認

1. 独立物理E-stopとFirmware watchdogの有無、責任者、停止手順を確認する。
2. 全6腱のEncoder方向とmm換算を実測し、Calibration hashを固定する。
3. 腱別Absolute Excursion範囲とIncrement上限を手動低速試験から承認する。
4. 腱別張力Zero、Soft候補、Hard limitを校正記録へ紐付ける。
5. MCP、PIP、DIPの個体別安全角度範囲を承認する。
6. Sensor欠損、通信断、NaN、Logger overflow、Operator StopをDry-runで再現する。
7. Safety停止後にReturnを自動実行しない設定を確認する。
8. PC書込完了と物理停止確認を区別してCheck listへ記載する。

## 推奨する順序

### 1. 静止記録

駆動せず、Servo、Encoder、張力、DLCの周期、鮮度、Drop、Zero driftを記録する。

### 2. 単腱小振幅Step

各腱について正負の小さなStepを実施し、Excursion応答、張力、停止時間、往復差を測る。閾値はこの結果から決める。

### 3. D0: Moment Arm FF

小変位の共通Goal集合で複数初期姿勢・複数反復を収集する。Raw Event完全性、Decision周期、Action保持、Reward Frame対応を先に確認する。

### 4. D1: FF+FB

D0と同じGoal、初期姿勢、Calibration、Safety設定を使い、最終誤差、整定、保持、最大張力、FB補正量を比較する。

### 5. D2: 制限付き探索

D1が安全に再現できた後だけ、単腱の小さな正負探索から始める。全6腱への独立Noiseは初期段階で使用しない。

## 次に収集するMetadata

- Experiment、Episode、Goal、Controller、D0/D1/D2 ID
- 実測初期姿勢、初期Excursion、初期張力
- CalibrationとSafety設定のhash
- requested、filtered、PID-reflectedのtargetとdelta
- Capture、Inference、Decision、PID、Transport時刻
- Logger drop、Transport receipt、物理適用状態
- Stop reason、Safety reasons、Operator操作

Online AWACは、新Schemaの連続Episode、固定Policy Mock検証、安全Review、固定Policy実機評価が完了するまで実行しない。

