import time
from datetime import datetime
import random
from io import StringIO
import requests

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
import streamlit as st

def display_message(role, message_text):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if role == 'human':
        message_class = 'chat-human'
        icon = '👩'  
        align = 'flex-start'  
    else:
        message_class = 'chat-bot'
        icon = '👩'  
        align = 'flex-end'  

    full_message = f"""
    <div class="chat-container {message_class}" style="justify-content: {align};">
        <div class="message">
            <span class="icon">{icon}</span>
            <p>{message_text}</p>
            <p class="timestamp">Sent at {timestamp}</p>
        </div>
    </div>
    <style>
    .chat-container {{
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
        width: 100%;
    }}
    .chat-human .message {{
        background-color: #00004B;  /* Dark color for human messages */
        border-radius: 10px;
        padding: 10px;
        margin-left: 10px;
        color: white;  /* Text color for better contrast */
        max-width: 70%;
    }}
    .chat-bot .message {{
        background-color: #00008B;  /* Blue color for AI messages */
        border-radius: 10px;
        padding: 10px;
        margin-right: 10px;
        color: white;  /* Text color for better contrast */
        max-width: 70%;
    }}
    .icon {{
        font-size: 24px;
        line-height: 1;
        display: block;
        margin-bottom: 5px;
    }}
    .timestamp {{
        font-size: 12px;
        color: gray;
        margin-top: 5px;
    }}
    </style>
    """
    st.markdown(full_message, unsafe_allow_html=True)

def responseTime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_for_response = len(result) * 3 / 20

        sleep_time = 0

        if elapsed_time < time_for_response:
            sleep_time = time_for_response - elapsed_time + random.uniform(0,3)
            time.sleep(sleep_time)

        return result, sleep_time

    return wrapper


@responseTime
def get_response(with_message_history, user_input):
    response = with_message_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "abc123"}},
    ).content
    return response


st.title("Lucy AI Girlfriend")
response_sys_prompt = requests.get('https://api.crazytweaks.online/aichat/sys_prompt.txt')
response_trigger_prompt = requests.get('https://api.crazytweaks.online/aichat/trigger_prompt.txt')

with st.sidebar:
    openai_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    #uploaded_file1 = st.file_uploader(
    #    "Upload system prompt file in txt format", type="txt"
    #)
    #uploaded_file2 = st.file_uploader(
    #    "Upload trigger prompt file in txt format", type="txt"
    #)
    real_response = st.toggle("Realistic response time")
    response_time = st.write('')

#if uploaded_file1 and uploaded_file2 and openai_key:
if openai_key:
    sys_prompt = response_sys_prompt.text
    trigger_prompt = response_trigger_prompt.text

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
        #display_message('human',user_input)
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

        if real_response:
            response, sleep_time = get_response(with_message_history, user_input)
        #timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #full_message = f"{response}\n*Sent at {timestamp}*"
            st.chat_message("ai").write(response)
            #display_message('ai', response)
            with st.sidebar:
                st.write(f"Response generated in: {sleep_time:.2f} seconds")           
        else:
            response = with_message_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "abc123"}},
            ).content
            st.chat_message("ai").write(response)
            #display_message('ai', response)
