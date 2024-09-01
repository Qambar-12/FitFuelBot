import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Function to interact with Groq API for athlete responses
def respond_athletes(user_input,multi_turn=False):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
    You are an expert dietitian, nutritionist, and doctor specializing in athletic training and performance enhancement. Your role is to provide comprehensive and tailored diet, nutrition, and workout routines based on the specific characteristics and goals of the athlete. The athlete details are as follows : 
     If ```{multi_turn}``` respond keeping in mind the previous messages from list 
    - Age: {user_input['age']}
    - Gender: {user_input['gender']}
    - Weight: {user_input['weight']} kg
    - Height: {user_input['height']} cm
    - Athlete Type: {user_input['athlete_type']}
    - Injury: {user_input['injury']}
    - Injury Details: {user_input['injury_details']}
    - Duration: {user_input['duration']}
    - Season: {user_input['season']}
    - Workout Type: {user_input['workout_type']}
    - Additional Details: {user_input['additional_details']}

    Return ```Invalid input``` if any of the above parameters are anomalous or incorrect followed by a brief explanation.
    Return ```Please provide all the necessary details``` if any of the above parameters are missing except for the injury details and additional details.
    Based on the above information, provide a detailed response that includes:

        1. **Customized Diet Plan (for age year old...gender...athlete type...)**:
       - Include daily caloric intake based on the athlete's age, gender, weight, height, and goals.
       - Provide a sample meal plan for a day with meals and snacks, detailing macronutrients (carbohydrates, proteins, fats) and micronutrients (vitamins, minerals).
       - Consider any dietary preferences or restrictions and suggest alternatives or substitutions where necessary.

    2. **Workout Routine**:
       - Suggest a weekly training plan based on the athlete's current fitness level and training frequency.
       - Include different types of workouts such as strength training, cardio, flexibility exercises, and sport-specific drills.
       - Provide guidelines on intensity, duration, and repetitions, adjusting for injury history and athletic discipline based on the season and workout type and duration.

    3. **Recovery and Injury Prevention**:
       - Offer advice on recovery strategies including rest, hydration, and sleep.
       - Suggest injury prevention techniques or exercises tailored to the athlete's injury history and goals.
       - Provide any additional recommendations to optimize performance and prevent future injuries.

    4. **General Health and Nutrition Tips**:
       - Provide overall health tips that align with the athlete's training and nutritional needs.
       - Include information on the importance of hydration, proper sleep, and mental health practices.

    Modify the response based on additional details Additional Details: {user_input['additional_details']} provided by user
    Format the response in a clear tabular format and organize each sections with clear headings. Use bullet points and numbered lists where appropriate for clarity to make sure all advice is specific, actionable, and tailored to the athlete's unique characteristics and objectives.
    """
            }
        ],
        model="llama3-70b-8192",
        temperature=0.5,
    )
    return chat_completion.choices[0].message.content


# Streamlit UI setup
st.title("FitFuelBot: Personalized Assistant")
st.write("Welcome to FitFuelBot! Select an assistant type to get started.")

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = None

# Dropdown for selecting the assistant type
assistant_type = st.sidebar.selectbox(
    "Select Your Assistant",
    ["Athlete Assistant", "Gym Assistant", "Medical Assistant"]
)

if assistant_type == "Gym Assistant" or assistant_type == "Medical Assistant":
    st.sidebar.write("**Feature Under Construction.** Please check back later.")
elif assistant_type == "Athlete Assistant":
    # Athlete form in the sidebar
    with st.sidebar.form(key='athlete_form'):
        gender = st.radio("Gender", ("Male", "Female", "Other"))
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
        athlete_type = st.selectbox("Type of Athlete", ["Runner", "Swimmer", "Footballer", "Cricketer", "Gymnast", "Basketball Player", "Tennis Player", "Other"])
        injury = st.radio("Do you have any injury?", ("No", "Yes"))
        injury_details = ""
        if injury == "Yes":
            injury_details = st.text_input("Please provide details of the injury")
        duration = st.slider("Time Duration (Months)", min_value=1, max_value=12, step=1)
        season = st.selectbox("Season", ["Off-Season", "Pre-Season", "In-Season"])
        workout_type = st.selectbox("Type of Workout", ["Beginner", "Intermediate", "Advanced"])
        additional_details = st.text_area("Additional Details", placeholder="Enter any additional information here...")

        submit_button = st.form_submit_button(label='Submit')

    # Handling form submission
    if submit_button:
        if st.session_state.user_input:
            st.session_state.conversation = []
            st.write('---')  # Divider
            st.session_state.initial_response = None
        user_input = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'athlete_type': athlete_type,
            'injury': injury,
            'injury_details': injury_details,
            'duration': duration,
            'season': season,
            'workout_type': workout_type,
            'additional_details': additional_details
        }
        st.session_state.user_input = user_input

        # Get the response from Groq API
        response = respond_athletes(user_input)

        if response and "Invalid input" not in response and "Please provide all the necessary details" not in response:
            st.session_state.initial_response = response
        else:
            st.session_state.user_input = None
            st.session_state.initial_response = None
            st.sidebar.error(f"Failed to generate a response (Invalid input)")

# Display initial response and start multi-turn conversation
if st.session_state.initial_response:
    st.chat_message("assistant").markdown(st.session_state.initial_response)
    st.write("---")  # Divider
    st.write("### Chat with FitFuelBot")

    # Multi-turn chat interface
    for msg in st.session_state.conversation:
        st.chat_message(msg['role']).markdown(msg['content'])
        st.write("---") 

    user_message = st.text_input("You:", key='user_message')
    if st.button("Send"):
        if user_message:
            # Append user message to conversation
            st.session_state.conversation.append({"role": "user", "content": user_message})

            # Generate a response from the bot
            st.session_state.user_input['additional_details'] += "\n" + user_message
            bot_response = respond_athletes(st.session_state.user_input, multi_turn=True)

            # Append bot response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": bot_response})

            # Display bot response
            st.chat_message("assistant").markdown(bot_response)
        else:
            st.error("Please enter a message.")
else:
    st.write("Please submit your details in the sidebar first to start the conversation.")
