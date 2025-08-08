from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import json


@dataclass
class OllamaTestResult:
    """Test result data class"""

    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class OllamaTestStats:
    """Test statistics"""

    def __init__(self):
        self.results: List[OllamaTestResult] = []
        self.start_time = datetime.now()

    def add_result(self, result: OllamaTestResult):
        self.results.append(result)

    def export_results(self, path: str = "test_results.json"):
        """Export test results to a JSON file
        Args:
            path: Output file path
        """
        results_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "total_duration": sum(r.duration for r in self.results),
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"\nTest results saved to: {path}")

    def print_summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        duration = sum(r.duration for r in self.results)

        print("\n=== Test Summary ===")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration:.2f} seconds")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for result in self.results:
                if not result.success:
                    print(f"- {result.name}: {result.error}")
