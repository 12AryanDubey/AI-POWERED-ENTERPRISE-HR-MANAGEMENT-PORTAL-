import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import plotly.express as px
import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2

# --- LOAD SECRETS & CONFIGURE AI ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Enterprise HR AI Portal",
    page_icon="👔",
    layout="wide"
)

st.title("👔 Enterprise HR Management & AI Portal")
st.markdown("---")

# --- LOAD TRAINED ML MODEL & SCALER ---
@st.cache_resource
def load_ml_components():
    with open('models/attrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_ml_components()

# --- TABS FOR DIFFERENT FEATURES ---
tab1, tab2, tab3 = st.tabs(["📊 Employee Attrition Predictor", "📁 Database Records", "🤖 HR Policy Chatbot"])

# ==========================================
# TAB 1: ATTRITION PREDICTOR
# ==========================================
with tab1:
    st.header("Predict Employee Attrition Risk")
    st.write("Enter employee metrics below to evaluate retention risk using the trained AI model.")

    col1, col2 = st.columns(2)

    with col1:
        emp_name = st.text_input("Employee Name", "John Doe")
        department = st.selectbox("Department", ["Sales", "Technical", "HR", "Marketing", "Finance"])
        satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.65, 0.05)
        last_eval = st.slider("Last Evaluation Score", 0.0, 1.0, 0.75, 0.05)
        salary_map = {"Low": 1, "Medium": 2, "High": 3}
        salary_str = st.selectbox("Salary Level", ["Low", "Medium", "High"], index=1)

    with col2:
        projects = st.number_input("Number of Projects Assigned", 1, 10, 4)
        monthly_hours = st.number_input("Average Monthly Hours", 50, 350, 180)
        tenure = st.number_input("Years at Company", 1, 20, 3)
        accident = st.selectbox("Work Accident History", ["No", "Yes"])
        promotion = st.selectbox("Promoted in Last 5 Years", ["No", "Yes"])

    if st.button("🔮 Calculate Risk Score", type="primary"):
        work_accident_val = 1 if accident == "Yes" else 0
        promotion_val = 1 if promotion == "Yes" else 0
        salary_val = salary_map[salary_str]

        features = np.array([[
            satisfaction, last_eval, projects, 
            monthly_hours, tenure, work_accident_val, 
            promotion_val, salary_val
        ]])

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0][1] * 100

        st.markdown("### Risk Analysis Results")
        
        c1, c2 = st.columns(2)
        with c1:
            if prediction == 1:
                st.error(f"⚠️ **High Attrition Risk**: {prob:.1f}% chance of leaving!")
            else:
                st.success(f"✅ **Low Attrition Risk**: {prob:.1f}% chance of leaving.")
        
        with c2:
            st.metric(label="Overall Risk Index", value=f"{prob:.1f}%")

        conn = sqlite3.connect('hr_database.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO employees (name, department, performance_score, attrition_risk) VALUES (?, ?, ?, ?)",
            (emp_name, department, float(last_eval), float(prob))
        )
        conn.commit()
        conn.close()
        st.toast(f"Record for '{emp_name}' saved to SQLite database!")

# ==========================================
# TAB 2: DATABASE VIEW
# ==========================================
with tab2:
    st.header("Saved Employee Records")
    
    conn = sqlite3.connect('hr_database.db')
    df_db = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()

    if not df_db.empty:
        st.dataframe(df_db, use_container_width=True)
        
        fig = px.bar(
            df_db, 
            x="name", 
            y="attrition_risk", 
            color="attrition_risk", 
            title="Attrition Risk Comparison by Employee",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records evaluated yet. Run a prediction in Tab 1 to save data!")

# ==========================================
# TAB 3: HR POLICY CHATBOT
# ==========================================
with tab3:
    st.header("🤖 HR Policy AI Assistant")
    
    @st.cache_data
    def get_pdf_text(filename):
        text = ""
        try:
            with open(filename, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            return None
        return text

    pdf_text = get_pdf_text("HR-Policy.pdf") 
    
    if pdf_text is None:
        st.error("❌ Could not find 'HR-Policy.pdf'. Please check the file name.")
    else:
        st.write("Ask any questions about the company HR policies (Leave, Attendance, Separation, etc.).")
        
        try:
            # THIS PULLS THE EXACT LIST OF MODELS GOOGLE ALLOWS FOR YOUR KEY
            valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            # ADDS A DROPDOWN TO YOUR WEB PAGE
            st.markdown("---")
            selected_model = st.selectbox("⚙️ **Select AI Model** (If one gives an error, just pick a different one!)", valid_models)
            st.markdown("---")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input("E.g., What are the working hours?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    try:
                        ai_model = genai.GenerativeModel(selected_model)
                        sys_prompt = f"You are a helpful company HR assistant. Use ONLY the following policy document to answer the user's question. If the answer is not in the document, say 'I cannot find that in the HR policy.'\n\nPOLICY DOCUMENT:\n{pdf_text}\n\nUSER QUESTION: {prompt}"
                        
                        response = ai_model.generate_content(sys_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error connecting to AI: {e}")
                        
        except Exception as e:
            st.error("Could not load Google models. Please check your API key.")