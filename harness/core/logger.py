import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .models import StageLog, Trajectory, StageName, BenchmarkResult


class StageLogger:

    def __init__(self, log_dir: str = "harness/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_trajectory_logs: List[StageLog] = []
        self.current_query_id: Optional[str] = None

    def start_trajectory(self, query_id: str):
        self.current_query_id = query_id
        self.current_trajectory_logs = []

    def log_stage(self, stage: StageName, input_data: Dict[str, Any], output_data: Dict[str, Any],
                  duration_ms: float = 0.0, metadata: Dict[str, Any] = None):
        log = StageLog(
            stage=stage,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        self.current_trajectory_logs.append(log)

        self._write_stage_log(log)

    def _write_stage_log(self, log: StageLog):
        if not self.current_query_id:
            return

        stage_dir = self.log_dir / self.current_query_id / log.stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)

        timestamp = log.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        log_file = stage_dir / f"{timestamp}.json"

        with open(log_file, 'w') as f:
            json.dump(log.to_dict(), f, indent=2)

    def get_trajectory_logs(self) -> List[StageLog]:
        return self.current_trajectory_logs

    def save_trajectory_summary(self, trajectory: Trajectory):
        if not self.current_query_id:
            return

        traj_dir = self.log_dir / self.current_query_id
        traj_dir.mkdir(parents=True, exist_ok=True)

        summary_file = traj_dir / "trajectory_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(trajectory.to_dict(), f, indent=2)

    def save_benchmark_result(self, result: BenchmarkResult):
        benchmark_dir = self.log_dir / result.benchmark / result.model_name
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = benchmark_dir / f"results_{timestamp}.json"

        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        latest_file = benchmark_dir / "latest_results.json"
        with open(latest_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)


class ConsoleLogger:

    @staticmethod
    def info(msg: str):
        print(f"[INFO] {msg}")

    @staticmethod
    def stage_start(stage: StageName, query_id: str):
        print(f"\n{'='*60}")
        print(f"STAGE: {stage.value.upper()} | Query: {query_id}")
        print(f"{'='*60}")

    @staticmethod
    def stage_complete(stage: StageName, duration_ms: float):
        print(f"[COMPLETE] {stage.value} ({duration_ms:.1f}ms)")

    @staticmethod
    def stage_output(stage: StageName, output_summary: str):
        print(f"[OUTPUT] {stage.value}: {output_summary[:200]}...")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"[WARNING] {msg}")