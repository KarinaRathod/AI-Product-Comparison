import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from crewai import Agent, Task, Crew, LLM

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# LLM
# -----------------------------
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Product Comparison PRO", layout="wide")
st.title("🛍️ AI Product Comparison PRO MAX")
st.caption("Compare multiple products with AI-powered insights")

# -----------------------------
# INPUT
# -----------------------------
st.sidebar.header("📥 Add Products")

num_products = st.sidebar.slider("Number of Products", 2, 5, 2)

products = []
prices = []
features = []

for i in range(num_products):
    st.sidebar.subheader(f"Product {i+1}")
    name = st.sidebar.text_input(f"Name {i+1}", key=f"name{i}")
    price = st.sidebar.number_input(f"Price {i+1}", min_value=0, key=f"price{i}")
    feat = st.sidebar.text_area(f"Features {i+1}", key=f"feat{i}")

    if name:
        products.append(name)
        prices.append(price)
        features.append(feat)

# -----------------------------
# TABLE
# -----------------------------
if products:
    st.subheader("📊 Product Table")

    df = pd.DataFrame({
        "Product": products,
        "Price": prices
    })

    st.dataframe(df)

# -----------------------------
# PRICE CHART
# -----------------------------
if len(products) > 1:
    st.subheader("📈 Price Comparison")

    fig, ax = plt.subplots()
    ax.bar(products, prices)
    ax.set_title("Price Comparison")

    st.pyplot(fig)

# -----------------------------
# SCORING SYSTEM
# -----------------------------
st.subheader("⚖️ Value Score")

scores = []
for i in range(len(products)):
    # simple scoring: more features + lower price = better
    feature_score = len(features[i].split(",")) if features[i] else 1
    score = feature_score / (prices[i] + 1)
    scores.append(score)

if scores:
    score_df = pd.DataFrame({
        "Product": products,
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    st.dataframe(score_df)

# -----------------------------
# AI COMPARISON
# -----------------------------
if st.button("🧠 AI Deep Comparison"):

    if len(products) < 2:
        st.warning("⚠️ Add at least 2 products")
        st.stop()

    comparator = Agent(
        role="Senior Product Analyst",
        goal="Compare products deeply and recommend best one",
        backstory="Expert in e-commerce and tech reviews",
        llm=llm
    )

    product_details = ""
    for i in range(len(products)):
        product_details += f"""
        Product: {products[i]}
        Price: {prices[i]}
        Features: {features[i]}
        """

    task = Task(
        description=f"""
        Compare these products:

        {product_details}

        Provide:
        - Feature comparison table
        - Pros & cons for each
        - Best value product
        - Best premium product
        - Final recommendation based on user needs
        """,
        expected_output="Detailed structured comparison",
        agent=comparator
    )

    crew = Crew(agents=[comparator], tasks=[task])
    result = crew.kickoff()

    st.subheader("💡 AI Recommendation")
    st.write(result.raw)

# -----------------------------
# BEST PRODUCT
# -----------------------------
if scores:
    best_index = scores.index(max(scores))
    st.success(f"🏆 Best Value Product: {products[best_index]}")

# -----------------------------
# DOWNLOAD REPORT
# -----------------------------
if st.button("⬇️ Download Report"):

    report = "PRODUCT COMPARISON REPORT\n\n"

    for i in range(len(products)):
        report += f"{products[i]} - ₹{prices[i]}\nFeatures: {features[i]}\n\n"

    st.download_button(
        "Download File",
        data=report,
        file_name="product_comparison.txt"
    )