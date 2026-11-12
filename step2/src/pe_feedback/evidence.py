"""Evidence extraction from public successful actions and their observations."""
import ast
from dataclasses import asdict, dataclass
import math

from .bridge import TOOLS_BY_NAME, parse_action
from .goal import Goal


@dataclass
class Fact:
    key: str
    value: object
    source_kind: str
    source_id: str
    active: bool = True


def bound_call(action):
    parsed = parse_action(action)
    spec = TOOLS_BY_NAME.get(parsed.name)
    kwargs = dict(parsed.kwargs)
    if spec:
        for arg, value in zip(spec.args, parsed.positional):
            kwargs.setdefault(arg.name, value)
    return parsed.name, kwargs


class Evidence:
    """Rebuild a small relevant ledger from public goal and executed history.

    An accepted action changes observed state; it does not redefine the goal.
    Irrelevant successful queries are not selected for feedback.
    """
    def __init__(self, goal_text, steps):
        self.goal = Goal.parse(goal_text)
        self.facts = {}
        self.completed = False
        self.deleted = False
        self.reminded = False
        self.ambiguity = False
        self.receipt_id = None
        self.receipt_source = None
        for key, value in self.goal.values.items():
            self.put("goal." + key, value, "user_goal", "goal")
        for index, step in enumerate(steps, 1):
            if step.ok:
                self.observe(step, index)

    def put(self, key, value, source_kind, source_id):
        self.facts[key] = Fact(key, value, source_kind, source_id)

    def get(self, key):
        fact = self.facts.get(key)
        return fact.value if fact and fact.active else None

    def invalidate(self, key):
        if key in self.facts:
            self.facts[key].active = False

    def observe(self, step, index):
        try:
            name, kw = bound_call(step.action)
        except Exception:
            return
        g, kind = self.goal.values, self.goal.kind
        source = f"obs:{index}"
        try:
            result = ast.literal_eval(step.observation)
        except (ValueError, SyntaxError):
            result = step.observation
        contacts = [result] if name == "find_contact" and isinstance(result, dict) else result if name == "list_contacts" and isinstance(result, list) else []
        for contact in contacts:
            if isinstance(contact, dict) and g.get("name", "").casefold() == str(contact.get("name", "")).casefold() and "email" in contact:
                self.put("contact_email", contact["email"], "tool_success", source)
        if kind == "roster" and name == "list_contacts" and kw.get("team", "").casefold() == g["team"].casefold() and isinstance(result, list):
            names = [c["name"] for c in result if isinstance(c, dict) and "name" in c]
            if names and len(names) == len(result):
                self.put("team_names", names, "tool_success", source)
        if kind == "email_file" and name == "read_file" and kw.get("path") == g["path"]:
            self.put("body", step.observation, "tool_success", source)
        if kind == "email_file" and name in {"write_file", "delete_file"} and kw.get("path") == g["path"]:
            self.invalidate("body")
        if kind == "cancel_event" and name == "list_events" and kw.get("date") == g["date"] and isinstance(result, list):
            matches = [e for e in result if isinstance(e, dict) and e.get("title") == g["title"] and e.get("date") == g["date"]]
            self.invalidate("event_id")
            self.ambiguity = len(matches) > 1
            if len(matches) == 1 and "id" in matches[0]:
                self.put("event_id", matches[0]["id"], "tool_success", source)
        if kind == "convert" and name == "convert_units":
            try:
                match = float(kw.get("value")) == float(g["value"]) and kw.get("from_unit") == g["from_unit"] and kw.get("to_unit") == g["to_unit"]
                if match and isinstance(result, (int, float)) and math.isfinite(result):
                    self.put("conversion_result", result, "tool_success", source)
            except (TypeError, ValueError):
                pass
        if kind == "email_file" and name == "send_message":
            email, body = self.get("contact_email"), self.get("body")
            if email is not None and body is not None and kw.get("to") == email and kw.get("subject") == g["subject"] and body in str(kw.get("body", "")):
                self.completed = True
                self.receipt_source = source
        if kind == "schedule" and name == "create_event":
            email = self.get("contact_email")
            if email is not None and all(kw.get(key) == g[key] for key in ("title", "date", "time")) and email in kw.get("attendees", []):
                self.completed = True
                self.receipt_id = step.observation.removeprefix("created ").strip()
                self.receipt_source = source
        if kind == "schedule" and name == "cancel_event" and kw.get("event_id") == self.receipt_id:
            self.completed = False
        if kind == "cancel_event" and name == "cancel_event" and self.get("event_id") is not None and kw.get("event_id") == self.get("event_id"):
            self.completed = True
            self.receipt_source = source
        if kind == "archive":
            if name == "delete_file" and kw.get("path") == g["path"]:
                self.deleted = True
                self.reminded = False  # Require a reminder after deletion.
            if name == "write_file" and kw.get("path") == g["path"]:
                self.deleted = False
            if name == "set_reminder" and kw.get("date") == g["date"] and kw.get("text") == g["text"] and self.deleted:
                self.reminded = True
                self.receipt_source = source
            self.completed = self.deleted and self.reminded
        if kind in {"roster", "convert"}:
            if name == "delete_file" and kw.get("path") == g["out_path"]:
                self.completed = False
            if name == "write_file" and kw.get("path") == g["out_path"]:
                content = str(kw.get("content", ""))
                if kind == "roster":
                    expected = self.get("team_names")
                    self.completed = expected is not None and sorted(s.strip() for s in content.split(",")) == sorted(expected)
                else:
                    expected = self.get("conversion_result")
                    try:
                        self.completed = expected is not None and abs(float(content.strip()) - expected) <= 0.00005
                    except ValueError:
                        self.completed = False
                if self.completed:
                    self.receipt_source = source
        if self.completed:
            self.put("completion_receipt", "Requirements observed satisfied", "tool_success", self.receipt_source)
        else:
            self.invalidate("completion_receipt")

    def snapshot(self):
        return {"goal_kind": self.goal.kind, "supported": self.goal.kind != "unsupported", "observed_complete": self.completed,
                "facts": [asdict(f) for f in self.facts.values()], "ambiguous_target": self.ambiguity}

    def progress_key(self):
        return {key: fact.value for key, fact in self.facts.items() if fact.active}, self.deleted, self.completed

    def next_step(self):
        g, kind = self.goal.values, self.goal.kind
        if self.completed:
            return "finish", "The requested requirements have observable completion receipts."
        if kind in {"email_file", "schedule"} and self.get("contact_email") is None:
            return "find_contact", f"Look up the requested person's full name: {g['name']}."
        if kind == "email_file":
            if self.get("body") is None:
                return "read_file", f"Read the requested source file {g['path']} to obtain its current full text."
            return "send_message", "Send the verified file text to the verified recipient, using the exact requested subject."
        if kind == "schedule":
            return "create_event", "Create the requested event using the verified attendee and the exact requested title, date and time."
        if kind == "cancel_event":
            if self.get("event_id") is None:
                return "list_events", f"Establish a unique event matching the requested title and date {g['date']} before cancelling."
            return "cancel_event", "Cancel the verified target event ID."
        if kind == "roster":
            if self.get("team_names") is None:
                return "list_contacts", f"Retrieve the members of the requested team {g['team']}."
            return "write_file", "Write all verified full names, comma separated, to the requested output path."
        if kind == "archive":
            if not self.deleted:
                return "delete_file", "Delete the file named in the request."
            return "set_reminder", "Set the requested reminder with the exact date and text."
        if kind == "convert":
            if self.get("conversion_result") is None:
                return "convert_units", "Obtain the conversion result for the requested value and unit pair."
            return "write_file", "Write only the verified numeric result to the requested output file."
        return None, "Use the user request and observed tool results to establish the next requirement."
