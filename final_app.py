import json
import re
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client assuming that the environment variable is set
groq_client = Groq()

def gym_app():
    
# # Load JSON data from a specific path
# @st.cache(allow_output_mutation=True)
    def load_data():
        try:
            with open(r"C:\Users\SHAUKAT ALI\Desktop\Projects\GENAI_HACKATHON\output_file1.json", 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            st.error("File not found.")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            return {}
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return {}


    # Load the data
    data = load_data()

    model = SentenceTransformer('all-MiniLM-L6-v2')


    def create_index(data):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [entry['diet_plan']['description'] + ' ' + ' '.join(entry['lifestyle_tips']) for entry in data['data']]
        embeddings = model.encode(texts)  # Generate embeddings for each text entry

        dimension = embeddings.shape[1]  # Dimension of embeddings
        faiss_index = faiss.IndexFlatL2(dimension)  # Flat index uses L2 distance for similarity
        faiss_index.add(embeddings)  # Adding embeddings to the index
        return faiss_index, model


    index = create_index(data)

    class RAGSystem:
        def __init__(self, index, model):
            self.index = index
            self.model = model
            self.data = data

        def retrieve(self, query, k=3):
            query_vector = self.model.encode([query])
            _, indices = self.index.search(query_vector, k)
            return [self.data['data'][i] for i in indices.flatten()]


    index, model = create_index(data)  # Create the index and get the model used for encoding
    rag_system = RAGSystem(index, model)  # Initialize the RAG system with index and model


    # Streamlit UI Setup and Functions
    def parse_height(height_str):
        if "cm" in height_str:
            return float(height_str.replace("cm", "").strip())
        else:
            feet, inches = map(float, re.findall(r'\d+\.?\d*', height_str))
            return (feet * 30.48) + (inches * 2.54)

    def calculate_bmi(weight, height_cm):
        return weight / ((height_cm / 100) ** 2)

    def generate_plan(user_data, retrieved_data):
        prompt = f"""Based on the following user data and retrieved information, generate a personalized 7-day diet plan, exercise routine, and lifestyle tips for {user_data['goal']} with a focus on {user_data['focus']}.

        User Data:
        - Height: {user_data['height_cm']}cm
        - Current Weight: {user_data['weight']}kg
        - Desired Weight: {user_data['desired_weight']}kg
        - Waist: {user_data['waist']}cm
        - Time Frame: {user_data['months']} months
        - BMI: {user_data['bmi']:.1f}
        - Goal: {user_data['goal']}
        - Focus: {user_data['focus']}
        - Additional Details: {user_data['additional_details']}

        Retrieved Information:
        {json.dumps(retrieved_data, indent=2)}

        Please provide:
        1. A list of ideal options for each meal (breakfast, lunch, and dinner) that can be chosen throughout the week. These options should be nutritionally balanced and cater to the user's dietary preferences and goals.
        2. A weekly exercise routine tailored to the user's goal and focus, categorized by specific muscle groups such as chest, back, biceps, triceps, legs, shoulders, and abs.
        3. Daily lifestyle tips for achieving optimal results.
        4. Any additional recommendations based on the user's specific situation and health needs.
        5. Workout plan should be in tabular form, with sets and reps as columns.
        6. Diet plan should be in tabular form with calorie breakdown in seperate column.

        Ensure the meal options are versatile and easy to prepare, allowing for flexibility throughout the week without being tied to specific days.
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable health and fitness advisor. Provide detailed, personalized advice based on the given information."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=2000,
        )

        return response.choices[0].message.content



    # Streamlit UI
    st.title("Personalized Health and Fitness Advisor")

    height_str = st.text_input("Enter your height (e.g., 180cm or 5ft 11in):")
    weight = st.number_input("Enter your current weight (kg):", min_value=30.0, max_value=300.0, value=70.0)
    desired_weight = st.number_input("Enter your desired weight (kg):", min_value=30.0, max_value=300.0, value=65.0)
    months = st.number_input("Enter the number of months for your goal:", min_value=1, max_value=24, value=3)
    waist = st.number_input("Enter your waist circumference (cm):", min_value=50.0, max_value=200.0, value=80.0)
    favorite_athlete = st.text_input("Enter your favorite athlete (optional):")

    goal = st.radio("Select your primary goal:", ["Weight Loss", "Weight Gain"])
    focus = st.radio("Select your training focus:", ["Bodybuilding", "Powerlifting/Strength Training"])

    additional_details = st.text_area("Enter any additional details or preferences:")

    if st.button("Generate Personalized Plan"):
        if height_str:
            height_cm = parse_height(height_str)
            bmi = calculate_bmi(weight, height_cm)

            user_data = {
                "height_cm": height_cm,
                "weight": weight,
                "desired_weight": desired_weight,
                "months": months,
                "waist": waist,
                "favorite_athlete": favorite_athlete,
                "goal": goal.lower(),
                "focus": focus.lower().replace("/", "_"),
                "bmi": bmi,
                "additional_details": additional_details
            }

            query = f"{goal} {focus} BMI:{bmi:.1f} {favorite_athlete}"
            retrieved_data = rag_system.retrieve(query)

            with st.spinner("Generating your personalized plan..."):
                plan = generate_plan(user_data, retrieved_data)

            st.subheader("Your Personalized Health and Fitness Plan")
            st.write(plan)
        else:
            st.error("Please enter your height.")





    st.caption("Disclaimer: This advice is AI-generated. Consult with healthcare professionals before making significant changes to your diet or exercise routine.")


def med_app():
    # Function to load JSON data from a specific path
    @st.cache_data
    def load_data():
        try:
            with open(r"C:\Users\SHAUKAT ALI\Desktop\Projects\GENAI_HACKATHON\otraining.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            st.error("File not found. Please check the path.")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            return {}
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return {}

    # Load the data
    data = load_data()

    model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_index(data):
        texts = []
        for entry in data.get('data', []):
            for day_key, day_info in entry.items():
                if isinstance(day_info, dict):
                    day_text = ' '.join([
                        day_info.get('breakfast', ''),
                        day_info.get('lunch', ''),
                        day_info.get('dinner', ''),
                        ' '.join(day_info.get('snacks', []))
                    ])
                    texts.append(day_text)

        embeddings = model.encode(texts)  # Generate embeddings for each text entry

        dimension = embeddings.shape[1]  # Dimension of embeddings
        faiss_index = faiss.IndexFlatL2(dimension)  # Flat index uses L2 distance for similarity
        faiss_index.add(embeddings)  # Adding embeddings to the index
        return faiss_index, model

    index, model = create_index(data)

    class RAGSystem:
        def __init__(self, index, model):
            self.index = index
            self.model = model
            self.data = data

        def retrieve(self, query, k=3):
            query_vector = self.model.encode([query])
            _, indices = self.index.search(query_vector, min(k, len(self.data['data'])))
        
        # Safeguard to ensure indices are within range
            results = []
            for idx in indices.flatten():
                if idx < len(self.data['data']):
                    results.append(self.data['data'][idx])
        
            return results


    rag_system = RAGSystem(index, model)  # Initialize the RAG system with index and model

    # Streamlit UI Setup and Functions
    def parse_height(height_str):
        if "cm" in height_str:
            return float(height_str.replace("cm", "").strip())
        else:
            feet, inches = map(float, re.findall(r'\d+\.?\d*', height_str))
            return (feet * 30.48) + (inches * 2.54)

    def calculate_bmi(weight, height_cm):
        return weight / ((height_cm / 100) ** 2)

    # Generating the personalized plan
    def generate_plan(user_data, retrieved_data):
        prompt = f"""Based on the following user data and retrieved information, generate a personalized 7 day workout plan that helps user in battling their disease, in a tabular form.: {user_data['disease']}...

        User Data:
        - Height: {user_data['height_cm']}cm
        - Current Weight: {user_data['weight']}kg
        - Age: {user_data['age']}years
        - BMI: {user_data['bmi']:.1f}
        - Disease/Condition: {user_data['disease']}
        - Additional Details: {user_data['additional_details']}

        Retrieved Information:
        {json.dumps(retrieved_data, indent=2)}

        Please provide:
        1. A 7-day workout plan .
        2. Foods to avoid.
        3. Daily lifestyle tips for optimal results.
        4. Couple of meal suggestions for breakfast, lunch, dinner. with calorie breakdown in fats, carbohydrates and protein in each meal. Show this is tabular form.

        Ensure the plan is realistic, sustainable, and tailored to the user's preferences and constraints.
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable nutritionist and workout planner. Give the user relevant response to their query."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    # Streamlit UI
    st.title("Personalized Health and Fitness Advisor")
    
   
    height_str = st.text_input("Enter your height (e.g., 180cm or 5ft 11in):")
    weight = st.number_input("Enter your current weight (kg):", min_value=30.0, max_value=300.0, value=70.0)
    age = st.number_input("Enter your age:",max_value=200,value=45)
    disease = st.selectbox("Select your disease/condition (if any):", ["None", "Diabetes", "Hypertension", "Lactose Intolerance", "Other"])

    # Conditional input for "Other disease"
    if disease == "Other":
        other_disease = st.text_input("Please specify your condition:")

    additional_details = st.text_area("Enter any additional details or preferences:")

    if st.button("Submit"):
        height_cm = parse_height(height_str)
        bmi = calculate_bmi(weight, height_cm)

        user_data = {
            "height_cm": height_cm,
            "weight": weight,
            "age": age,
            "bmi": bmi,
            "disease": disease if disease != "Other" else other_disease,
            "additional_details": additional_details
        }


        query = f"Condition: {user_data['disease']} BMI:{bmi:.1f}"
        retrieved_data = rag_system.retrieve(query)

        with st.spinner("Generating your personalized plan..."):
            plan = generate_plan(user_data, retrieved_data)

        st.subheader("Your Personalized Health and Fitness Plan")
        st.write(plan)
    else:
        st.error("Please enter your height.")

    st.caption("Disclaimer: This advice is AI-generated. Consult with healthcare professionals before making significant changes to your diet or exercise routine.")
def ath_app():
    st.write("Wait")
    
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

    submit_button = st.button('Submit')

# Handling form submission
    if submit_button:
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

        # Get the response from Groq API
        response = respond_athletes(user_input)
        st.write(response)
        
st.title("FITFUEL BOT")
st.write("Welcome to your journey to getting fit.")    

st.sidebar.header("Select any one option:")
model_choice = st.sidebar.selectbox(
    "Which model do you want to use?",
    ("Gym Assistant", "Athlete Assistant", "Medical Assistant")
)
if model_choice == "Gym Assistant":
    gym_app()
elif model_choice == "Medical Assistant":
    med_app()
elif model_choice == "Athlete Assistant":
    ath_app();        