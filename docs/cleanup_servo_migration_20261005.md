# サーボ実装の一時整理（2026-10-05）

## 変更内容

削除専用ブランチで、次の旧実装・ローカル依存を削除する。

- `src/servo/`
- `src/pca9685/`
- `third_party/pyftdi/`
- `src/controller/pastSMAFinger/`（現在の作業ツリーにだけ存在する Git 管理外のローカルディレクトリ）

`src/servo/` の削除により不要になる、旧サーボ実装に依存する次のテストも削除する。

- `test/test_reliable_position_control.py`
- `test/test_multiturn_position_control.py`
- `test/test_verify_six_servo_position_control.py`
- `test/test_verify_six_servo_5hz_stream.py`
- `test/test_synchronized_feedback.py`

`pyproject.toml` から削除済みの `third_party/pyftdi` を package discovery 対象から外す。

## 影響

現時点では、`src/controller` のサーボ制御・位置制御・CSV再生・検証スクリプトが旧 `servo` パッケージを import しているため、これらの実行経路は一時的に使用できない。

本ブランチには `third_party/ServoHardwareAPI` はまだ Git 管理対象として含まれていない。そのため、削除後にサーボ機能を戻すには、同 API の追加または配置を行ったうえで、少なくとも次を再構築する必要がある。

- 速度・位置・マルチターン位置の指令 API
- Servo ID の変換（旧実装の `0`〜`5` と物理 ID の対応）
- テレメトリ形式の変換と到着確認
- `src/controller` の import と呼び出し
- 削除したサーボ関連テストの再作成または移植

既存の `src/controller` に追加した `TEMPORARY MIGRATION` コメントは、再構築後に該当コードとともに削除する。

## 後処理

`third_party/ServoHardwareAPI` を使ったサーボ機能の修正が完了し、関連テストと実機確認が通ったら、このドキュメント自体を削除する。
