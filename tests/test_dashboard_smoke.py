"""Smoke test: execute dashboard/app.py top-to-bottom against a fake `streamlit`
module (widgets return their default values). Catches Python errors in the
dashboard without a browser. Uses the real Streamlit only for `streamlit run`.
"""
import runpy
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _CM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSt(types.ModuleType):
    """Minimal stand-in for the streamlit API used by the dashboard."""

    def __init__(self, name="streamlit", press_buttons=True):
        super().__init__(name)
        self.calls = []
        self.session_state = {}
        self.press_buttons = press_buttons
        self.sidebar = self if name == "streamlit" else None

    # decorators
    def cache_data(self, func=None, **kw):
        return func if func else (lambda f: f)

    cache_resource = cache_data

    # layout
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [FakeSt("col", self.press_buttons) for _ in range(n)]

    def tabs(self, names):
        return [_CM() for _ in names]

    def expander(self, *a, **k):
        return _CM()

    def spinner(self, *a, **k):
        return _CM()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    # widgets: return defaults
    def selectbox(self, label, options, index=0, **k):
        return list(options)[index]

    def multiselect(self, label, options, default=None, **k):
        return list(default) if default is not None else []

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value

    def select_slider(self, label, options=None, value=None, **k):
        return value

    def checkbox(self, label, value=False, **k):
        return value

    def number_input(self, label, min_value=None, max_value=None, value=0, **k):
        return value

    def button(self, *a, **k):
        return self.press_buttons

    def pyplot(self, fig, **k):
        assert fig is not None
        self.calls.append("pyplot")

    def line_chart(self, data, **k):
        assert len(data) > 0
        self.calls.append("line_chart")

    def dataframe(self, data, **k):
        self.calls.append("dataframe")

    def __getattr__(self, name):
        # title, header, markdown, metric, table, image, json, caption, info, ...
        def f(*a, **k):
            if name == "image":
                assert Path(str(a[0])).exists(), a[0]
            return None
        return f


@contextmanager
def fake_streamlit():
    fake = FakeSt()
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake
    try:
        yield fake
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            del sys.modules["streamlit"]


class TestDashboard(unittest.TestCase):
    def test_app_runs(self):
        with fake_streamlit() as st:
            runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
        self.assertIn("pyplot", st.calls)
        self.assertIn("line_chart", st.calls)


if __name__ == "__main__":
    unittest.main()
