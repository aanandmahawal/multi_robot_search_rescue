"""RescueSim (multi-robot search & rescue simulator) on Streamlit.

    streamlit run streamlit_app.py

The 3-D dashboard (rescue/web) runs unchanged as a Streamlit component. Instead of calling the
web server of ``python -m rescue serve``, the page sends every request (reset, step, camera image,
comparison...) as the component's value; Streamlit reruns this script, which answers with the same
code the web server uses (rescue.server.Session) and hands the answer back as the component's
argument. Each browser tab keeps its own mission in st.session_state.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import streamlit as st
import streamlit.components.v1 as components

from rescue.config import RescueConfig
from rescue.server import MAX_STEPS, Compare, Session, _cfg_from, _clean

WEB = Path(__file__).parent / "rescue" / "web"
BASE = RescueConfig()

st.set_page_config(page_title="RescueSim · Multi-Robot Search & Rescue Simulator", page_icon="🚨", layout="wide",
                   initial_sidebar_state="collapsed")
# the dashboard has its own layout: give it the whole window
st.markdown("""<style>
  header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
  .block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stAppViewContainer"], .stApp { background: #090d13; }
  iframe { display: block; border: 0; }
</style>""", unsafe_allow_html=True)

dashboard = components.declare_component("rescue_dashboard", path=str(WEB))


@st.cache_resource
def comparisons() -> Compare:
    """One pool of worker processes for the algorithm comparisons, shared by every visitor.
    Two workers by default: a free Streamlit Community Cloud app has little memory (RESCUE_WORKERS overrides)."""
    return Compare(int(os.environ.get("RESCUE_WORKERS", 2)))


def _query(params: dict) -> dict:
    """The page's settings in the form the web server reads them (as from a URL)."""
    return parse_qs(urlencode({k: v for k, v in params.items() if v is not None}))


def _session() -> Session:
    if "mission" not in st.session_state:
        st.session_state.mission = Session(BASE)
    return st.session_state.mission


def handle(method: str, p: dict):
    """Answer one request of the page, exactly as rescue/server.py does over HTTP."""
    if method == "reset":
        s = _session()
        s.reset(_cfg_from(_query(p), BASE))
        return {"world": s.world_payload(), "state": s.state_payload()}
    s = _session()
    plan = p.get("plan")
    plan = int(plan) if isinstance(plan, (int, float)) else None
    if method == "step":
        for _ in range(max(1, min(MAX_STEPS, int(p.get("n", 1))))):
            s.sim.step()
        return s.state_payload(plan)
    if method == "state":
        return s.state_payload(plan)
    if method == "reveal":
        s.reveal = True
        return s.state_payload()
    if method == "camera":
        png = s.camera_png(int(p.get("robot", 0)), p.get("kind", "both"))
        return "data:image/png;base64," + base64.b64encode(png).decode()
    if method == "victim_map":
        return s.victim_map()
    if method == "compare":
        q = dict(p)
        vary, fast = q.pop("vary", "planner"), str(q.pop("fast", "1")) == "1"
        return {"job": comparisons().start(_cfg_from(_query(q), BASE), vary, fast)}
    if method == "compare_status":
        return comparisons().status(p["job"])
    raise ValueError(f"unknown request {method!r}")


@st.fragment
def mission_control():
    """The page's latest request is the component's value; answer it once. A fragment: a request
    reruns only this function, not the whole script, which keeps the round trip short."""
    request = st.session_state.get("rescue")
    if request and request.get("id") != st.session_state.get("answered"):
        st.session_state.answered = request["id"]
        try:
            result = json.loads(json.dumps(_clean(handle(request["method"], request.get("params") or {})), allow_nan=False))
            st.session_state.response = {"id": request["id"], "ok": True, "result": result}
        except Exception as e:                                     # shown on the page, like the web server's errors
            st.session_state.response = {"id": request["id"], "ok": False, "error": f"{type(e).__name__}: {e}"}
    dashboard(response=st.session_state.get("response"), key="rescue", default=None, height=860)


mission_control()
