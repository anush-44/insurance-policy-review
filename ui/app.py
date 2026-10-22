"""PolicyLens Streamlit UI — main application.

Run with:
    streamlit run ui/app.py
"""

import httpx
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyLens — Health Insurance Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

/* Cards / containers */
.policy-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: box-shadow 0.3s ease;
}
.policy-card:hover {
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
}

/* Citation pill */
.citation-pill {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 3px;
}

/* Red flag severity badges */
.badge-high   { background:#ef4444; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-medium { background:#f97316; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-low    { background:#22c55e; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }

/* Confidence bar */
.confidence-bar {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #ef4444, #f97316, #22c55e);
    margin-top: 6px;
}

/* Answer box */
.answer-box {
    background: rgba(99,102,241,0.15);
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    color: #e2e8f0;
    line-height: 1.7;
}

/* Abstention warning */
.abstain-box {
    background: rgba(239,68,68,0.15);
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    color: #fca5a5;
}

/* Verdict box */
.verdict-better { background: rgba(34,197,94,0.15); border-left: 4px solid #22c55e; border-radius:0 12px 12px 0; padding:0.75rem 1rem; }
.verdict-worse  { background: rgba(239,68,68,0.15);  border-left: 4px solid #ef4444;  border-radius:0 12px 12px 0; padding:0.75rem 1rem; }
.verdict-same   { background: rgba(99,102,241,0.15); border-left: 4px solid #6366f1;  border-radius:0 12px 12px 0; padding:0.75rem 1rem; }

/* Section header accent */
.section-accent {
    border-bottom: 2px solid #6366f1;
    padding-bottom: 6px;
    margin-bottom: 1rem;
    color: #a5b4fc;
    font-size: 1.1rem;
    font-weight: 600;
}

/* Disclaimer */
.disclaimer {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #fbbf24;
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
DISCLAIMER = (
    "⚠️ **Disclaimer:** This tool is for informational purposes only and does not "
    "constitute insurance advice. Always refer to the official policy wording document."
)


# ── Helper: API calls ─────────────────────────────────────────────────────────
def api_get(path: str, **params) -> dict | list | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("❌ Cannot reach the PolicyLens API. Start the server with:\n```\nuvicorn app.main:app --reload\n```")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_post(path: str, json: dict | None = None, data: dict | None = None, files=None) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, data=data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("❌ Cannot reach the PolicyLens API.")
        return None
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        st.error(f"API error {e.response.status_code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def get_policies() -> list[dict]:
    result = api_get("/api/v1/policies")
    return result if isinstance(result, list) else []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 PolicyLens")
    st.markdown("*AI-Powered Health Insurance Analyser*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📤 Ingest Policy", "💬 Ask Questions", "📋 Policy Summary", "🚩 Red Flags", "⚖️ Compare"],
        label_visibility="collapsed",
    )
    st.divider()

    policies = get_policies()
    if policies:
        st.markdown("**📂 Indexed Policies**")
        for p in policies:
            st.markdown(
                f"<div class='policy-card' style='padding:0.6rem 0.9rem; margin-bottom:0.4rem;'>"
                f"<b style='color:#a5b4fc'>{p['policy_name'][:30]}</b><br>"
                f"<span style='color:#94a3b8; font-size:0.8rem'>{p['insurer']} · {p['total_chunks']} chunks</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No policies indexed yet. Upload one to get started.")

    st.divider()
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Ingest Policy
# ══════════════════════════════════════════════════════════════════════════════
if page == "📤 Ingest Policy":
    st.markdown("# 📤 Ingest a Policy")
    st.markdown("Upload a health insurance policy PDF to index it for Q&A and analysis.")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"], key="upload_pdf")
    with col2:
        policy_id = st.text_input("Policy ID (optional)", placeholder="hdfc_optima_secure")
        insurer = st.text_input("Insurer Name", value="Unknown Insurer")

    if uploaded and st.button("🚀 Ingest Policy", type="primary"):
        with st.spinner("Parsing, embedding, and indexing… this may take 1–2 minutes."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            data = {"insurer": insurer}
            if policy_id:
                data["policy_id"] = policy_id
            resp = api_post("/api/v1/policies/ingest", data=data, files=files)

        if resp:
            st.success(f"✅ **{resp['policy_name']}** indexed successfully!")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Policy ID", resp["policy_id"])
            col_b.metric("Total Pages", resp["total_pages"])
            col_c.metric("Chunks Indexed", resp["total_chunks"])
            st.balloons()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Ask Questions
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Ask Questions":
    st.markdown("# 💬 Ask About Your Policy")
    st.markdown("Get plain-language answers grounded in exact policy clauses.")
    st.divider()

    if not policies:
        st.warning("Please ingest a policy first.")
        st.stop()

    policy_options = {f"{p['policy_name']} ({p['insurer']})": p["policy_id"] for p in policies}

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox("Select Policy", list(policy_options.keys()))
    with col2:
        top_k = st.slider("Context chunks", 3, 15, 5)

    question = st.text_area(
        "Your Question",
        placeholder="Is knee replacement surgery covered? What is the waiting period for pre-existing diseases?",
        height=80,
    )

    # Example questions
    st.markdown("**Try these:**")
    example_cols = st.columns(3)
    examples = [
        "What is the waiting period for pre-existing diseases?",
        "Is maternity covered? If so, what are the conditions?",
        "What is the room rent limit per day?",
        "Are AYUSH treatments covered?",
        "What is the co-pay percentage?",
        "List the major exclusions in this policy.",
    ]
    for i, eq in enumerate(examples):
        if example_cols[i % 3].button(eq, key=f"eq_{i}", use_container_width=True):
            question = eq

    if st.button("🔍 Get Answer", type="primary", disabled=not question):
        pid = policy_options[selected_label]
        with st.spinner("Retrieving and generating grounded answer…"):
            resp = api_post(
                "/api/v1/qa",
                json={"question": question, "policy_id": pid, "top_k": top_k},
            )

        if resp:
            st.divider()
            conf = resp.get("confidence", 0.0)

            if resp.get("is_abstention"):
                st.markdown(
                    f"<div class='abstain-box'>🚫 <b>I couldn't find this in the policy.</b><br>"
                    f"{resp['answer']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='answer-box'>{resp['answer']}</div>",
                    unsafe_allow_html=True,
                )

            # Confidence indicator
            conf_color = "#22c55e" if conf >= 0.7 else "#f97316" if conf >= 0.4 else "#ef4444"
            st.markdown(
                f"**Confidence:** {conf:.0%} "
                f"<span style='color:{conf_color}'>{'●'*int(conf*10)}{'○'*(10-int(conf*10))}</span>",
                unsafe_allow_html=True,
            )

            # Citations
            citations = resp.get("citations", [])
            if citations:
                st.markdown("#### 📎 Source Citations")
                for cit in citations:
                    with st.expander(
                        f"📄 Clause {cit['clause_id']} · Page {cit['page']}", expanded=False
                    ):
                        st.markdown(f"> *{cit['quote']}*")

            # Caveats
            caveats = [c for c in resp.get("caveats", []) if c and "disclaimer" not in c.lower()]
            if caveats:
                for cv in caveats:
                    st.warning(cv)

            st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Policy Summary
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Policy Summary":
    st.markdown("# 📋 Policy Summary Card")
    st.markdown("Auto-extracted overview of key policy terms.")
    st.divider()

    if not policies:
        st.warning("Please ingest a policy first.")
        st.stop()

    policy_options = {f"{p['policy_name']} ({p['insurer']})": p["policy_id"] for p in policies}
    selected_label = st.selectbox("Select Policy", list(policy_options.keys()))

    if st.button("📋 Extract Summary", type="primary"):
        pid = policy_options[selected_label]
        with st.spinner("Analysing policy clauses and extracting summary…"):
            resp = api_get(f"/api/v1/analyse/summary/{pid}")

        if resp:
            st.success("Summary extracted!")

            # Key metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PED Waiting Period", resp.get("waiting_period_ped", "N/A"))
            c2.metric("Room Rent Cap", resp.get("room_rent_cap", "N/A"))
            c3.metric("Co-pay", resp.get("copay_percent", "None"))
            c4.metric("Specific Illness Wait", resp.get("waiting_period_specific", "N/A"))

            st.divider()

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("<div class='section-accent'>🔄 Restoration Benefit</div>", unsafe_allow_html=True)
                st.write(resp.get("restoration_benefit", "Not specified"))

                st.markdown("<div class='section-accent'>🎁 No-Claim Bonus</div>", unsafe_allow_html=True)
                st.write(resp.get("no_claim_bonus", "Not specified"))

                st.markdown("<div class='section-accent'>💊 Sub-Limits</div>", unsafe_allow_html=True)
                sublimits = resp.get("key_sublimits", [])
                if sublimits:
                    for sl in sublimits:
                        st.markdown(f"• {sl}")
                else:
                    st.write("None specified")

            with col_right:
                st.markdown("<div class='section-accent'>🚫 Key Exclusions</div>", unsafe_allow_html=True)
                exclusions = resp.get("key_exclusions", [])
                if exclusions:
                    for ex in exclusions:
                        st.markdown(f"❌ {ex}")
                else:
                    st.write("None specified")

            st.divider()
            st.markdown("<div class='section-accent'>📝 Claim Process</div>", unsafe_allow_html=True)
            st.write(resp.get("claim_process_summary", "Not specified"))

            st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Red Flags
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚩 Red Flags":
    st.markdown("# 🚩 Red Flag Scanner")
    st.markdown("Identifies buyer-unfriendly, hidden, or restrictive policy clauses.")
    st.divider()

    if not policies:
        st.warning("Please ingest a policy first.")
        st.stop()

    policy_options = {f"{p['policy_name']} ({p['insurer']})": p["policy_id"] for p in policies}
    selected_label = st.selectbox("Select Policy", list(policy_options.keys()))

    if st.button("🔎 Scan for Red Flags", type="primary"):
        pid = policy_options[selected_label]
        with st.spinner("Scanning exclusion and restriction clauses…"):
            resp = api_get(f"/api/v1/analyse/red-flags/{pid}")

        if resp:
            flags = resp.get("red_flags", [])
            if not flags:
                st.success("✅ No major red flags detected!")
            else:
                high = [f for f in flags if f.get("severity") == "high"]
                medium = [f for f in flags if f.get("severity") == "medium"]
                low = [f for f in flags if f.get("severity") == "low"]

                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 High Severity", len(high))
                c2.metric("🟠 Medium Severity", len(medium))
                c3.metric("🟢 Low Severity", len(low))
                st.divider()

                for flag in sorted(flags, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("severity", "low"), 3)):
                    sev = flag.get("severity", "medium")
                    badge_class = f"badge-{sev}"
                    with st.expander(
                        f"{'🔴' if sev=='high' else '🟠' if sev=='medium' else '🟢'} "
                        f"{flag.get('flag_type', 'Issue')} — Clause {flag.get('clause_id', '?')} · Page {flag.get('page', '?')}",
                        expanded=sev == "high",
                    ):
                        st.markdown(
                            f"<span class='{badge_class}'>{sev.upper()}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(flag.get("description", ""))

            st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Compare Policies
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare":
    st.markdown("# ⚖️ Policy Comparison")
    st.markdown("Side-by-side comparison of two policies for a specific health scenario.")
    st.divider()

    if len(policies) < 2:
        st.warning("Please ingest at least two policies to compare.")
        st.stop()

    policy_options = {f"{p['policy_name']} ({p['insurer']})": p["policy_id"] for p in policies}
    policy_labels = list(policy_options.keys())

    col1, col2 = st.columns(2)
    with col1:
        label_a = st.selectbox("Policy A", policy_labels, index=0)
    with col2:
        label_b = st.selectbox("Policy B", policy_labels, index=min(1, len(policy_labels) - 1))

    scenario_presets = [
        "knee replacement surgery",
        "maternity and newborn care",
        "pre-existing diabetes treatment",
        "cancer chemotherapy",
        "mental health hospitalisation",
        "AYUSH treatment",
        "road traffic accident emergency",
    ]
    scenario_label = st.selectbox(
        "Scenario (or type your own below)",
        ["Custom…"] + scenario_presets,
    )
    if scenario_label == "Custom…":
        scenario = st.text_input("Describe the scenario", placeholder="cataract surgery")
    else:
        scenario = scenario_label

    if label_a == label_b:
        st.warning("Please select two different policies.")
    elif st.button("⚖️ Compare Now", type="primary", disabled=not scenario):
        pid_a = policy_options[label_a]
        pid_b = policy_options[label_b]
        with st.spinner(f"Comparing policies for '{scenario}'…"):
            resp = api_post(
                "/api/v1/analyse/compare",
                json={"scenario": scenario, "policy_id_a": pid_a, "policy_id_b": pid_b},
            )

        if resp:
            st.success(f"Comparison complete for **{scenario}**")
            st.divider()

            # Overall verdict
            st.markdown(
                f"<div class='answer-box'><b>Overall Verdict:</b> {resp.get('overall_verdict', '')}</div>",
                unsafe_allow_html=True,
            )

            # Comparison table
            points = resp.get("comparison_points", [])
            if points:
                st.markdown("### Aspect-by-Aspect Breakdown")
                for pt in points:
                    verdict = pt.get("verdict", "")
                    verdict_class = (
                        "verdict-better" if "A is better" in verdict
                        else "verdict-worse" if "B is better" in verdict
                        else "verdict-same"
                    )
                    with st.expander(f"📌 {pt.get('aspect', 'Aspect')}", expanded=True):
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown(f"**{resp['policy_a_name'][:30]}**")
                            st.markdown(pt.get("policy_a_summary", ""))
                            if pt.get("policy_a_clause_id"):
                                st.caption(
                                    f"📄 Clause {pt['policy_a_clause_id']} · Page {pt.get('policy_a_page', '?')}"
                                )
                        with col_right:
                            st.markdown(f"**{resp['policy_b_name'][:30]}**")
                            st.markdown(pt.get("policy_b_summary", ""))
                            if pt.get("policy_b_clause_id"):
                                st.caption(
                                    f"📄 Clause {pt['policy_b_clause_id']} · Page {pt.get('policy_b_page', '?')}"
                                )
                        st.markdown(
                            f"<div class='{verdict_class}'><b>Verdict:</b> {verdict}</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)
