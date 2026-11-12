"""Public-goal parsing. No Task, payload, reference actions, or world access."""
from dataclasses import dataclass
import re


@dataclass
class Goal:
    kind: str
    values: dict
    text: str

    @classmethod
    def parse(cls, text):
        patterns = [
            ("email_file", r"Send (?P<name>.+?) the contents of (?P<path>\S+)\. Use '(?P<subject>[^']+)' as the subject\."),
            ("cancel_event", r"Cancel the '(?P<title>[^']+)' meeting scheduled on (?P<date>\d{4}-\d{2}-\d{2})\."),
            ("schedule", r"Schedule a 1:1 with (?P<name>.+?) on (?P<date>\d{4}-\d{2}-\d{2}) at (?P<time>\d{2}:\d{2})\. Title the event '(?P<title>[^']+)'\."),
            ("roster", r"Write the names of everyone on the (?P<team>.+?) team, comma separated, to (?P<out_path>\S+)\."),
            ("archive", r"Delete (?P<path>\S+), then set a reminder for (?P<date>\d{4}-\d{2}-\d{2}) saying '(?P<text>[^']+)'\."),
            ("convert", r"Convert (?P<value>[\d.]+) (?P<from_unit>\S+) to (?P<to_unit>\S+) and write only the resulting number to (?P<out_path>\S+)\."),
        ]
        for kind, pattern in patterns:
            match = re.fullmatch(pattern, text)
            if match:
                return cls(kind, match.groupdict(), text)
        return cls("unsupported", {}, text)

