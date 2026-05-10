import streamlit as st
import os
from agent import graph

st.set_page_config(page_title="Deep Research Agent", layout="wide")

st.title("Deep Research Agent 🔍")
st.markdown("Enter a company or topic below to generate a comprehensive, structured research report using parallel AI sub-agents.")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

company = st.text_input("Target Company / Topic", placeholder="e.g. Anthropic, SpaceX, Stripe")

if st.button("Start Research") and company:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        st.info(f"Starting deep research on **{company}**...")

        # Containers for real-time progress
        progress_container = st.container()
        report_container = st.container()

        with progress_container:
            st.subheader("Progress")
            status_text = st.empty()
            vectors_list = st.empty()
            findings_expander = st.expander("Live Findings", expanded=True)

        initial_state = {"company": company, "plan": None, "findings": [], "report": None}

        status_text.text("Phase 1: Planning...")

        final_state = None

        try:
            # We use stream to show progress
            for event in graph.stream(initial_state):
                if "planner_node" in event:
                    status_text.text("Phase 1 Complete. Phase 2: Orchestrating parallel agents...")
                    plan = event["planner_node"].get("plan")
                    if plan:
                        vectors = [v.title for v in plan.vectors]
                        vectors_list.markdown("**Research Vectors:**\n" + "\n".join([f"- {v}" for v in vectors]))

                if "parallel_execution_node" in event:
                    status_text.text("Phase 2: Sub-Agents are researching vectors in parallel...")
                    status_text.text("Received finding from a sub-agent!")
                    node_data = event["parallel_execution_node"]
                    if "findings" in node_data:
                        for f in node_data["findings"]:
                            findings_expander.markdown(f"**Vector {f.vector_id}:**\n{f.markdown_content}\n---")

                if "synthesizer_node" in event:
                    status_text.text("Phase 3: Synthesizing final report...")
                    final_state = event["synthesizer_node"]

            status_text.success("Research Complete!")

            if final_state and "report" in final_state:
                with report_container:
                    st.header("Final Report")
                    st.markdown(final_state["report"])

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
