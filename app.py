import os
import chainlit as cl
from agent import graph

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Deep Research Agent 🔍\nEnter a company or topic below to generate a comprehensive, structured research report using parallel AI sub-agents. Please ensure your `OPENAI_API_KEY` is set in the environment.").send()

@cl.on_message
async def main(message: cl.Message):
    if not os.environ.get("OPENAI_API_KEY"):
        await cl.Message(content="⚠️ Error: OPENAI_API_KEY environment variable is not set. Please set it to proceed.").send()
        return

    company = message.content.strip()

    await cl.Message(content=f"Starting deep research on **{company}**...").send()

    initial_state = {"company": company, "plan": None, "findings": [], "report": None}

    final_state = None

    try:
        config = {"configurable": {"thread_id": "chainlit_session"}}

        # We stream the graph events to give real-time feedback
        async for event in graph.astream(initial_state, config=config):
            if "planner_node" in event:
                plan = event["planner_node"].get("plan")
                if plan:
                    vectors = [v.title for v in plan.vectors]
                    plan_msg = "**Phase 1 Complete. Research Plan Proposed:**\n" + "\n".join([f"- {v}" for v in vectors])
                    await cl.Message(content=plan_msg).send()

        # The graph pauses after the planner_node due to interrupt_after
        # Ask user for approval
        res = await cl.AskActionMessage(
            content="Do you approve this research plan?",
            actions=[
                cl.Action(name="approve", value="yes", label="✅ Approve & Continue"),
                cl.Action(name="reject", value="no", label="❌ Reject")
            ]
        ).send()

        if res and res.get("value") == "yes":
            await cl.Message(content="Plan approved! Proceeding to Phase 2 (Parallel Execution)...").send()

            # Resume graph execution
            async for event in graph.astream(None, config=config):
                if "parallel_execution_node" in event:
                    node_data = event["parallel_execution_node"]
                    if "findings" in node_data:
                        for f in node_data["findings"]:
                            finding_msg = f"**Received finding from sub-agent for Vector {f.vector_id}:**\n{f.markdown_content}"
                            await cl.Message(content=finding_msg).send()

                if "synthesizer_node" in event:
                    await cl.Message(content="**Phase 3:** All vectors researched. Synthesizing final report...").send()
                    final_state = event["synthesizer_node"]

            if final_state and "report" in final_state:
                await cl.Message(content=f"### Final Report\n\n{final_state['report']}").send()
        else:
            await cl.Message(content="Plan rejected. Research aborted.").send()

    except Exception as e:
        await cl.Message(content=f"❌ An error occurred during execution: {e}").send()
