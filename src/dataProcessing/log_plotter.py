# this script is for plotting results csv player

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 設定
# ============================================================

INPUT_FILE = Path(r"C:\\Users\\ynaga\\Desktop\\Finger-Control\\src\\dataProcessing\\log.txt")

OUTPUT_IMAGE = Path("all_tendons_position_difference.png")
OUTPUT_CSV = Path("all_tendons_position_difference.csv")

TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]


# ============================================================
# ログ解析
# ============================================================

def parse_playback_log(file_path: Path) -> pd.DataFrame:
    """
    PLAYBACK形式のログファイルから、時刻および各腱の
    target/actual位置を抽出する。

    Returns
    -------
    pd.DataFrame
        以下のような列を持つDataFrame:
        t,
        FDP_target, FDP_actual,
        FDS_target, FDS_actual,
        ...
    """

    if not file_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="replace")

    # 1行全体から時刻と target/actual 部分を抽出
    row_pattern = re.compile(
        r"PLAYBACK.*?"
        r"t=(?P<t>\d+(?:\.\d+)?)s.*?"
        r"target/actual\(error\):\s*(?P<data>.*)"
    )

    # 例:
    # FDP(servo 5)=718/718(+0)
    # FDP(servo 5)=718/N/A
    tendon_pattern = re.compile(
        r"(?P<tendon>FDP|FDS|ED|DI|PI|LUM)"
        r"\(servo\s+\d+\)="
        r"(?P<target>-?\d+)/"
        r"(?P<actual>N/A|-?\d+)"
    )

    rows = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        row_match = row_pattern.search(line)

        if row_match is None:
            continue

        row = {
            "t": float(row_match.group("t")),
        }

        tendon_matches = tendon_pattern.finditer(row_match.group("data"))

        for match in tendon_matches:
            tendon = match.group("tendon")
            target_text = match.group("target")
            actual_text = match.group("actual")

            row[f"{tendon}_target"] = float(target_text)

            if actual_text == "N/A":
                row[f"{tendon}_actual"] = float("nan")
            else:
                row[f"{tendon}_actual"] = float(actual_text)

        # 6本すべてのtargetが存在する行のみ採用
        required_columns = [
            f"{tendon}_target"
            for tendon in TENDONS
        ]

        if all(column in row for column in required_columns):
            rows.append(row)
        else:
            print(
                f"警告: {line_number}行目は必要なデータが不足しているため、"
                "スキップしました。"
            )

    if not rows:
        raise ValueError(
            "ログからPLAYBACKデータを抽出できませんでした。"
            "ログ形式を確認してください。"
        )

    df = pd.DataFrame(rows)

    # 時刻順に並べる
    df = df.sort_values("t").reset_index(drop=True)

    return df


# ============================================================
# 初期位置からの差分を計算
# ============================================================

def calculate_position_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    各target/actual系列について、最初に取得できた値を
    初期位置として差分を計算する。

    target:
        通常は t=0 の値が基準。

    actual:
        t=0 が N/A の場合、最初に取得できたactual値が基準。
    """

    result = pd.DataFrame({
        "t": df["t"],
    })

    for tendon in TENDONS:
        for data_type in ["target", "actual"]:
            column = f"{tendon}_{data_type}"

            if column not in df.columns:
                print(f"警告: {column} が存在しません。")
                result[column] = float("nan")
                continue

            valid_values = df[column].dropna()

            if valid_values.empty:
                print(f"警告: {column} に有効な値がありません。")
                result[column] = float("nan")
                continue

            initial_value = valid_values.iloc[0]
            result[column] = df[column] - initial_value

            print(
                f"{column}: 初期値 = {initial_value:g}"
            )

    return result


# ============================================================
# プロット
# ============================================================

def plot_all_tendons(
    difference_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    tendonごとに1つのグラフを作成し、
    targetとactualを重ねて表示する。
    """

    fig, axes = plt.subplots(
        nrows=len(TENDONS),
        ncols=1,
        figsize=(12, 18),
        sharex=True,
    )

    for ax, tendon in zip(axes, TENDONS):
        target_column = f"{tendon}_target"
        actual_column = f"{tendon}_actual"

        ax.plot(
            difference_df["t"],
            difference_df[target_column],
            label="Target",
            linewidth=1.8,
        )

        ax.plot(
            difference_df["t"],
            difference_df[actual_column],
            label="Actual",
            linewidth=1.8,
        )

        ax.axhline(
            y=0,
            linewidth=0.8,
        )

        ax.set_ylabel(
            f"{tendon}\nposition difference"
        )

        ax.grid(
            visible=True,
            alpha=0.3,
        )

        ax.legend(
            loc="best",
        )

    axes[-1].set_xlabel("t [s]")

    fig.suptitle(
        "All tendon position changes from initial values",
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    raw_df = parse_playback_log(INPUT_FILE)

    difference_df = calculate_position_differences(raw_df)

    difference_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    plot_all_tendons(
        difference_df,
        OUTPUT_IMAGE,
    )

    print()
    print("処理が完了しました。")
    print(f"抽出データ数: {len(raw_df)}")
    print(
        f"時刻範囲: "
        f"{raw_df['t'].min():.3f} ～ "
        f"{raw_df['t'].max():.3f} s"
    )
    print(f"グラフ: {OUTPUT_IMAGE.resolve()}")
    print(f"CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
