#!/bin/bash
set -e
pytest /tests/test_outputs.py -v --json-report --json-report-file=/logs/verifier/report.json
python3 -c "
import json
report = json.load(open('/logs/verifier/report.json'))
passed = report['summary'].get('passed', 0)
total = report['summary'].get('total', 1)
reward = 1 if passed == total else 0
open('/logs/verifier/reward.txt', 'w').write(str(reward))
"
