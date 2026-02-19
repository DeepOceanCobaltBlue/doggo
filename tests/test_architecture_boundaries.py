from __future__ import annotations

import unittest
from pathlib import Path


class ArchitectureBoundaries(unittest.TestCase):
    def test_config_app_does_not_directly_import_simulation_engine(self) -> None:
        src = Path("config_app.py").read_text()
        self.assertNotIn("from sim_core.engine import SimulationEngine", src)
        self.assertNotIn("SimulationEngine(", src)

    def test_command_runner_is_mode_agnostic(self) -> None:
        src = Path("motion_core/command_runner.py").read_text()
        # config_app owns UI mode strings; runner accepts output_target only.
        self.assertNotIn('"test"', src)
        self.assertNotIn("'test'", src)


if __name__ == "__main__":
    unittest.main()
