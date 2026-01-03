from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

load_dotenv()

from chains import generation_chain, reflection_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


def main():
    print("Hello from langchain-course-reflection-agent!")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
                        — newly Tool Calling feature is seriously underrated.

                        After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

                        Made a video covering their newest blog post

                        """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
