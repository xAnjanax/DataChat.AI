import streamlit as st 
from streamlit_lottie import st_lottie
import requests

def app():
    st.title("Welcome to DataChat.AI!")
    st.text("Your conversational assistant for analyzing data")
        
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_url_home = "https://lottie.host/f6616f8c-4eff-4c3c-97ee-3058bd1bc46d/czL31kov8g.json"
    lottie_hello = load_lottieurl(lottie_url_home)

    lottie_url_home1 = "https://lottie.host/78d2a831-2de7-4d17-8b1a-aa82aa463bc9/092nBN58Y4.json"
    lottie_hello1 = load_lottieurl(lottie_url_home1)

    st_lottie(
    lottie_hello,
    height=350,
    width=None,
    key=None,
    )

    st.text(" ")

    st.write('''### *Rough Guidelines for navigating this application:*''')
    st.write('''#### *Step 1: Upload your CSV file on the Home Page.*''') 
    st.write('''A preview of the data will be provided for your reference.''') 
    st.write('''You can upload more than one file, although only one file at a time. These files can be accessed in the Chatbot and Plot Pages.''')
    
    st.write('''#### *Step 2: Head over to the Chatbot Page.*''') 
    st.write('''Select the file you would like to work with.''') 
    st.write('''Here you can ask the chatbot about the contents of the dataset, and ask it for recommendations on how the data can be analyzed and plotted.''')

    st.write('''#### *Step 3: Head over to the Plot Page.*''') 
    st.write('''The file you selected on the Chatbot Page will be applied here.''') 
    st.write('''Utilize the interface to plot visualizations as suggested by the Chatbot. The information you generate on this page can be used for further queries.''') 
    st.write('''You can ask the chatbot for more suggestions as per your needs.''')

    st_lottie(
    lottie_hello1,
    height=330,
    width=1290,
    key=None,
    )

    text = "Happy Chatting/Plotting!"
    st.markdown(
        f"<h4 style='text-align: center;'>{text}</h4>",
        unsafe_allow_html=True
    )

    # st.write('''##### Happy Chatting/Plotting!''')
    
    st_lottie(
    lottie_hello1,
    height=350,
    width=None,
    key=None,
    )

    st.header("Upload a file to get started.")
