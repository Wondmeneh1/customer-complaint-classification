
import streamlit as st
import ollama

st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="📩"
)

st.title("Customer Complaint Classification")
st.write("Powered by Llama 3.2 and Ollama")

complaint = st.text_area(
    "Enter a customer complaint:"
)

if st.button("Classify Complaint"):
    if not complaint.strip():
        st.warning("Please enter a complaint.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
Classify the complaint into exactly one category:
Billing, Technical Issue, Delivery, Upgrade, Cancellation, Complaint, or Other.

Complaint: {complaint}

Return only the category name.
"""
                        }
                    ],
                    options={"temperature": 0}
                )

                st.success("Classification complete")
                st.write(
                    "**Predicted category:**",
                    response["message"]["content"]
                )

            except Exception as e:
                st.error(
                    "Could not connect to Ollama. "
                    "Check that Ollama is running."
                )
