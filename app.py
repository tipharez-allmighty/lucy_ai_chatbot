import time
import random
from io import StringIO

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
import streamlit as st


def responseTime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_for_response = 20 - elapsed_time

        if time_for_response > 0:
            time.sleep(random.uniform(1, time_for_response))

        return result

    return wrapper


@responseTime
def get_response(with_message_history, user_input):
    response = with_message_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "abc123"}},
    ).content
    return response


st.title("Lucy AI Girlfriend")

with st.sidebar:
    openai_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    uploaded_file1 = st.file_uploader(
        "Upload system prompt file in txt format", type="txt"
    )
    uploaded_file2 = st.file_uploader(
        "Upload trigger prompt file in txt format", type="txt"
    )

if uploaded_file1 and uploaded_file2 and openai_key:
    sys_prompt = StringIO(uploaded_file1.getvalue().decode("utf-8")).read()
    trigger_prompt = StringIO(uploaded_file2.getvalue().decode("utf-8")).read()

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=488,
        verbose=True,
        timeout=None,
        max_retries=2,
        api_key=openai_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    trigger_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", trigger_prompt),
            (MessagesPlaceholder(variable_name="history")),
        ]
    )

    runnable = prompt | model
    trigger_eval = trigger_prompt | model

    msgs = StreamlitChatMessageHistory(key="special_app_key")

    with_message_history = RunnableWithMessageHistory(
        runnable,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
    )

    if "trigger" not in st.session_state:
        st.session_state.trigger = None

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_input := st.chat_input("Whats up?"):
        if not openai_key:
            st.stop()

        st.chat_message("human").write(user_input)
        conversation_history = [
            {"type": msg.type, "content": msg.content} for msg in msgs.messages
        ]
        st.session_state.trigger = trigger_eval.invoke(
            {"history": conversation_history}
        ).content
        with st.sidebar:
            if str(st.session_state.trigger) == "0":
                st.write("No clear match yet")
            if str(st.session_state.trigger) == "1":
                st.write("Red Light")
            if str(st.session_state.trigger) == "2":
                st.write("Green Light")

        response = get_response(with_message_history, user_input)

        st.chat_message("human").write(response)
