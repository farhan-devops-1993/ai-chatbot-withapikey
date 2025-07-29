"""
Simple LangChain Streamlit App with Groq
A beginner-friendly version focusing on core concepts
"""

import streamlit as st  # import streamlit for web app    
from langchain.chat_models import init_chat_model  # import init_chat_model from langchain
from langchain_groq import ChatGroq  # import ChatGroq from langchain_groq
from langchain_core.output_parsers import StrOutputParser  # import StrOutputParser from langchain_core.output_parsers
from langchain_core.messages import HumanMessage, AIMessage  # import HumanMessage, AIMessage from langchain_core.messages
from langchain.prompts import ChatPromptTemplate  # import ChatPromptTemplate from langchain.prompts
import os  # import os for environment variables

# Page config
st.set_page_config(page_title="Simple LangChain Chatbot with Groq", page_icon="🚀")  # set page config

# Title
st.title("🚀 Simple LangChain Chat with Groq")  # set title
st.markdown("Learn LangChain basics with Groq's ultra-fast inference!")  # set subtitle

# Add left side panel
with st.sidebar:  # using sidebar to add left side things
    st.header("Settings")  # add a header

    # API Key input
    api_key = st.text_input("GROQ API Key", type="password", help="GET Free API Key at console.groq.com")  # password will not be visible

    # Model selection dropdown
    model_name = st.selectbox(
        "Model",
        ["llama3-8b-8192", "gemma2-9b-it"],  # available models
        index=0  # default model
    )

    # Clear Chat button
    if st.button("Clear Chat"):  # if clear chat button is clicked
        st.session_state.messages = []  # clear session state messages
        st.rerun()  # rerun the app

# Initialize chat history
if "messages" not in st.session_state:  # if messages not in session state
    st.session_state.messages = []  # start with empty chat

# Initialize LLM
@st.cache_resource  # cache this function so it's not called unnecessarily
def get_chain(api_key, model_name):  # api_key and model_name are user inputs
    if not api_key:  # if API key is missing
        return None  # return None if API key not provided

    # Initialize the Groq model
    llm = ChatGroq(
        groq_api_key=api_key,  # user's API key
        model_name=model_name,  # selected model
        temperature=0.7,  # creativity level
        streaming=True  # enable response streaming
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([  # building prompt structure
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),  # system instruction
        ("user", "{question}")  # placeholder for user input
    ])

    # Create LangChain chain
    chain = prompt | llm | StrOutputParser()  # process flow: prompt → model → output parser

    return chain  # return the constructed chain

# Get the chain
chain = get_chain(api_key, model_name)  # call the function to get chain

# If chain not ready, ask user to enter key
if not chain:  # if no valid chain
    st.warning("👆 Please enter your Groq API key in the sidebar to start chatting!")  # show warning
    st.markdown("[Get your free API key here](https://console.groq.com)")  # helpful link

else:
    # Display the chat messages (if chain is ready)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # display each message by role
            st.write(message["content"])  # print the content

    # Chat input field
    if question := st.chat_input("Ask me anything"):  # wait for user input
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})  # store user input
        with st.chat_message("user"):  # show user message in chat box
            st.write(question)

        # Generate response from model
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # placeholder for streamed response
            full_response = ""  # collect streamed chunks

            try:
                # Stream response from Groq
                for chunk in chain.stream({"question": question}):  # stream the model output
                    full_response += chunk  # append each chunk
                    message_placeholder.markdown(full_response + "▌")  # show partial response

                message_placeholder.markdown(full_response)  # show full response

                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})  # save response

            except Exception as e:  # error handling
                st.error(f"Error: {str(e)}")  # show error message

# Example questions section
st.markdown("---")  # horizontal rule
st.markdown("### 💡 Try these examples:")  # examples header
col1, col2 = st.columns(2)  # two-column layout for examples
with col1:
    st.markdown("- What is LangChain?")  # example 1
    st.markdown("- Explain Groq's LPU technology")  # example 2
with col2:
    st.markdown("- How do I learn programming?")  # example 3
    st.markdown("- Write a haiku about AI")  # example 4

# Footer
st.markdown("---")  # horizontal rule
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")  # footer note
