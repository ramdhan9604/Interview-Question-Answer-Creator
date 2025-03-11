import streamlit as st
import pandas as pd
from helper import llm_pipeline
import os

def main():
    st.set_page_config(page_title="PDF Question Answer Generator Chatbot")
    st.title("📄 PDF Question Answer Generator Chatbot")
    
    # Custom CSS for word wrapping
    st.markdown("""
        <style>
        .dataframe tbody tr th, .dataframe tbody tr td {
            white-space: normal !important;
            word-wrap: break-word;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = os.path.join("temp", uploaded_file.name)
        
        # Ensure the temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully!")
        
        if st.button("Generate Questions and Answers"):
            st.info("Processing the PDF and generating questions with answers...")
            
            # Run the LLM pipeline
            answer_generation_chain, filtered_ques_list = llm_pipeline(file_path)
            
            # Generate answers for each question
            answers = [answer_generation_chain.run({'query': q}).replace("\n", " ") for q in filtered_ques_list]

            
            # Convert questions and answers to DataFrame
            df = pd.DataFrame({"Questions": filtered_ques_list, "Answers": answers})
            
            # Display the dataframe with word wrapping
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
            
            # Provide download button for CSV file
            csv_file = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Questions and Answers as CSV",
                data=csv_file,
                file_name="generated_questions_answers.csv",
                mime="text/csv"
            )
            
            st.success("Questions and Answers generated successfully!")

if __name__ == "__main__":
    main()