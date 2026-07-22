from __future__ import annotations

import unittest

from controller.csv_player.excursion_player import ExcursionPlayer


def make_player(*, allow_out_of_range_positions: bool = False) -> ExcursionPlayer:
    return ExcursionPlayer(
        servo_ids=(5, 4, 2, 3, 1, 0),
        position_units_per_mm=(1, 1, 1, 1, 1, 1),
        allow_out_of_range_positions=allow_out_of_range_positions,
    )


class ExcursionPlayerPositionRangeTest(unittest.TestCase):
    def test_single_turn_player_rejects_out_of_range_position(self) -> None:
        player = make_player()

        with self.assertRaises(ValueError):
            player.build_command_frames(
                (0.0, 0.1),
                ((0, 0, 0, 0, 0, 0), (-1, 4096, 0, 0, 0, 0)),
                (0, 0, 0, 0, 0, 0),
            )

    def test_accumulated_player_preserves_out_of_range_position(self) -> None:
        player = make_player(allow_out_of_range_positions=True)

        frames = player.build_command_frames(
            (0.0, 0.1),
            ((0, 0, 0, 0, 0, 0), (-1, 4096, 0, 0, 0, 0)),
            (0, 0, 0, 0, 0, 0),
        )

        self.assertEqual(frames[-1].positions[:2], (-1, 4096))


if __name__ == "__main__":
    unittest.main()

