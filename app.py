import json
import re
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import faiss
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Initialize Groq client assuming that the environment variable is set
groq_client = Groq()

# # Load JSON data from a specific path
# @st.cache(allow_output_mutation=True)
def load_data():
    try:
        with open('C:\\Users\\musta\\Downloads\\output_file1.json', 'r', encoding='utf-8-sig') as file:
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

# # Check and print a small part of the data to ensure it's loaded correctly
# if data:
#     st.write("Metadata:", data['metadata'])
#     st.write("First entry in data:", data['data'][0])  # Displaying the first entry to check the content

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

# # Generating the personalized plan
# def generate_plan(user_data, retrieved_data):
#     prompt = f"""Based on the following user data and retrieved information, generate a personalized 7-day diet plan, exercise routine, and lifestyle tips for {user_data['goal']} with a focus on {user_data['focus']}...

#     User Data:
#     - Height: {user_data['height_cm']}cm
#     - Current Weight: {user_data['weight']}kg
#     - Desired Weight: {user_data['desired_weight']}kg
#     - Waist: {user_data['waist']}cm
#     - Time Frame: {user_data['months']} months
#     - BMI: {user_data['bmi']:.1f}
#     - Goal: {user_data['goal']}
#     - Focus: {user_data['focus']}
#     - Additional Details: {user_data['additional_details']}

#     Retrieved Information:
#     {json.dumps(retrieved_data, indent=2)}

#     Please provide:
#     1. A 7-day diet plan with specific meal suggestions.
#     2. A weekly exercise routine tailored to the user's goal and focus.
#     3. Daily lifestyle tips for optimal results.
#     4. Any additional recommendations based on the user's specific situation.

#     Ensure the plan is realistic, sustainable, and tailored to the user's preferences and constraints.
#     """

#     response = groq_client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a knowledgeable health and fitness advisor. Provide detailed, personalized advice based on the given information."},
#             {"role": "user", "content": prompt}
#         ],
#         model="llama3-70b-8192",
#         temperature=0.7,
#         max_tokens=2000,
#     )

#     return response.choices[0].message.content

# Generating the personalized plan
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
