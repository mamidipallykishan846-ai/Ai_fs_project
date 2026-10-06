import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import chromadb
import ollama

from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# SMART AGRICULTURAL SYSTEM
# Single-file Streamlit Application
# ============================================================

st.set_page_config(
    page_title="Smart Agricultural System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Smart Agricultural System"
DB_PATH = "./agriculture_chroma_db"
COLLECTION_NAME = "agriculture_knowledge"

DOCUMENT_DIR = Path("./agriculture_documents")
DOCUMENT_DIR.mkdir(exist_ok=True)


# ============================================================
# SAMPLE AGRICULTURAL DATA
# ============================================================

CROP_DATA = [
    {
        "Crop": "Rice",
        "Season": "Kharif",
        "Soil": "Clay",
        "pH_Min": 5.5,
        "pH_Max": 7.0,
        "Temp_Min": 20,
        "Temp_Max": 35,
        "Water": "High",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Wheat",
        "Season": "Rabi",
        "Soil": "Loamy",
        "pH_Min": 6.0,
        "pH_Max": 7.5,
        "Temp_Min": 10,
        "Temp_Max": 25,
        "Water": "Medium",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Maize",
        "Season": "Kharif",
        "Soil": "Loamy",
        "pH_Min": 5.5,
        "pH_Max": 7.0,
        "Temp_Min": 18,
        "Temp_Max": 32,
        "Water": "Medium",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Cotton",
        "Season": "Kharif",
        "Soil": "Black Soil",
        "pH_Min": 5.5,
        "pH_Max": 8.0,
        "Temp_Min": 21,
        "Temp_Max": 35,
        "Water": "Medium",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Groundnut",
        "Season": "Kharif",
        "Soil": "Sandy Loam",
        "pH_Min": 6.0,
        "pH_Max": 7.5,
        "Temp_Min": 20,
        "Temp_Max": 30,
        "Water": "Medium",
        "Fertilizer": "Phosphorus"
    },
    {
        "Crop": "Potato",
        "Season": "Rabi",
        "Soil": "Loamy",
        "pH_Min": 5.0,
        "pH_Max": 6.5,
        "Temp_Min": 15,
        "Temp_Max": 25,
        "Water": "Medium",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Tomato",
        "Season": "Rabi",
        "Soil": "Loamy",
        "pH_Min": 6.0,
        "pH_Max": 7.0,
        "Temp_Min": 18,
        "Temp_Max": 30,
        "Water": "Medium",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Sugarcane",
        "Season": "Annual",
        "Soil": "Loamy",
        "pH_Min": 6.0,
        "pH_Max": 7.5,
        "Temp_Min": 20,
        "Temp_Max": 35,
        "Water": "High",
        "Fertilizer": "NPK"
    },
    {
        "Crop": "Chickpea",
        "Season": "Rabi",
        "Soil": "Loamy",
        "pH_Min": 6.0,
        "pH_Max": 8.0,
        "Temp_Min": 15,
        "Temp_Max": 30,
        "Water": "Low",
        "Fertilizer": "Phosphorus"
    },
    {
        "Crop": "Millet",
        "Season": "Kharif",
        "Soil": "Sandy Loam",
        "pH_Min": 5.5,
        "pH_Max": 7.5,
        "Temp_Min": 20,
        "Temp_Max": 35,
        "Water": "Low",
        "Fertilizer": "Nitrogen"
    }
]

crop_df = pd.DataFrame(CROP_DATA)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return client, collection


embedding_model = load_embedding_model()
chroma_client, collection = load_chroma()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_pdf_text(file_path):
    """Extract text from PDF."""

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=800):
    """Split document text into chunks."""

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def add_document_to_database(file_path, filename):
    """Read PDF and store its embeddings in ChromaDB."""

    text = extract_pdf_text(file_path)

    if not text.strip():
        return 0

    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = [
        {
            "filename": filename,
            "source": "uploaded_pdf"
        }
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def search_knowledge(question, number_results=5):
    """Search agricultural knowledge base."""

    if collection.count() == 0:
        return []

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(
            number_results,
            collection.count()
        )
    )

    documents = result.get(
        "documents",
        [[]]
    )

    if not documents:
        return []

    return documents[0]


def ask_ai(question, context, model):
    """Ask Ollama agricultural AI assistant."""

    prompt = f"""
You are an expert Smart Agriculture AI assistant.

Your job is to help farmers with:

- Crop selection
- Soil management
- Irrigation
- Fertilizers
- Crop diseases
- Agricultural practices
- Farming technology
- General agricultural questions

Use the retrieved agricultural information below.

IMPORTANT:
1. Give simple and practical answers.
2. Do not invent information.
3. If the provided information is insufficient,
   clearly state that.
4. Give precautions when necessary.
5. Do not claim to replace an agricultural expert.
6. For pesticide/fertilizer quantities, recommend
   checking the product label or local agricultural
   extension officer.

RETRIEVED INFORMATION:
{context}

FARMER QUESTION:
{question}

Provide a clear answer.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def recommend_crops(
    soil,
    season,
    ph,
    temperature,
    water
):
    """Recommend crops based on farm conditions."""

    exact = crop_df[
        (crop_df["Soil"] == soil)
        &
        (
            (crop_df["Season"] == season)
            |
            (crop_df["Season"] == "Annual")
        )
        &
        (crop_df["pH_Min"] <= ph)
        &
        (crop_df["pH_Max"] >= ph)
        &
        (crop_df["Temp_Min"] <= temperature)
        &
        (crop_df["Temp_Max"] >= temperature)
        &
        (crop_df["Water"] == water)
    ]

    return exact


def calculate_crop_score(
    row,
    soil,
    season,
    ph,
    temperature,
    water
):
    """Calculate a compatibility score."""

    score = 0

    if row["Soil"] == soil:
        score += 30

    if row["Season"] == season:
        score += 20
    elif row["Season"] == "Annual":
        score += 15

    if row["pH_Min"] <= ph <= row["pH_Max"]:
        score += 20

    if row["Temp_Min"] <= temperature <= row["Temp_Max"]:
        score += 20

    if row["Water"] == water:
        score += 10

    return score


def get_ranked_crops(
    soil,
    season,
    ph,
    temperature,
    water
):
    """Return crops ranked by suitability."""

    data = crop_df.copy()

    data["Score"] = data.apply(
        lambda row: calculate_crop_score(
            row,
            soil,
            season,
            ph,
            temperature,
            water
        ),
        axis=1
    )

    return data.sort_values(
        "Score",
        ascending=False
    )


def generate_fertilizer_advice(
    crop,
    soil,
    ph
):
    """Basic fertilizer guidance."""

    crop_row = crop_df[
        crop_df["Crop"] == crop
    ]

    if crop_row.empty:
        return "No fertilizer information available."

    fertilizer = crop_row.iloc[0]["Fertilizer"]

    return f"""
### 🌱 Fertilizer Recommendation

**Crop:** {crop}

**Soil:** {soil}

**Soil pH:** {ph}

**Suggested fertilizer type:** {fertilizer}

Use soil-test results to determine the actual
application rate. Follow the fertilizer label
and local agricultural recommendations.
"""


def irrigation_advice(crop, water):
    """Basic irrigation recommendation."""

    if water == "Low":
        advice = """
- Prefer drought-tolerant crops.
- Avoid unnecessary irrigation.
- Use mulching to reduce evaporation.
- Consider drip irrigation where practical.
"""
    elif water == "Medium":
        advice = """
- Maintain regular irrigation according to crop stage.
- Avoid waterlogging.
- Check soil moisture before irrigation.
- Drip irrigation can improve water efficiency.
"""
    else:
        advice = """
- Ensure adequate water availability.
- Avoid excessive irrigation and waterlogging.
- Monitor soil moisture regularly.
- Consider efficient irrigation scheduling.
"""

    return f"""
### 💧 Irrigation Advice

**Crop:** {crop}

**Water availability:** {water}

{advice}
"""


# ============================================================
# HEADER
# ============================================================

st.title("🌱 Smart Agricultural System")

st.markdown(
    """
**AI-powered agriculture platform for crop recommendation,
soil management, irrigation guidance, agricultural document
search, and farmer assistance.**
"""
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌾 Navigation")

page = st.sidebar.radio(
    "Choose Module",
    [
        "🏠 Dashboard",
        "🌾 Crop Recommendation",
        "🤖 AI Agricultural Assistant",
        "📚 Agricultural Knowledge Base",
        "💧 Irrigation Advisor",
        "🧪 Fertilizer Advisor",
        "📊 Crop Database"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌾 Crops",
            len(crop_df)
        )

    with col2:
        st.metric(
            "📚 Knowledge Chunks",
            collection.count()
        )

    with col3:
        st.metric(
            "🤖 AI",
            "Ollama"
        )

    with col4:
        st.metric(
            "🧠 Embeddings",
            "MiniLM"
        )

    st.divider()

    st.subheader("🌱 System Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.markdown(
            """
            ### 🌾 Crop Recommendation

            - Soil-based crop selection
            - pH compatibility
            - Temperature compatibility
            - Season compatibility
            - Water requirement matching
            - Crop suitability score
            """
        )

        st.markdown(
            """
            ### 💧 Irrigation

            - Water availability analysis
            - Irrigation suggestions
            - Water-saving practices
            - Soil moisture awareness
            """
        )

    with feature_col2:

        st.markdown(
            """
            ### 🤖 AI Assistant

            - Natural language questions
            - Agricultural knowledge retrieval
            - PDF-based knowledge
            - Local Ollama AI
            """
        )

        st.markdown(
            """
            ### 🧪 Farm Management

            - Fertilizer recommendations
            - Soil information
            - Crop database
            - Agricultural document management
            """
        )

    st.divider()

    st.subheader("📈 Available Crops")

    st.dataframe(
        crop_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CROP RECOMMENDATION
# ============================================================

elif page == "🌾 Crop Recommendation":

    st.header("🌾 AI Crop Recommendation")

    st.info(
        "Enter your farm conditions to identify suitable crops."
    )

    col1, col2 = st.columns(2)

    with col1:

        soil = st.selectbox(
            "🌍 Soil Type",
            sorted(
                crop_df["Soil"].unique()
            )
        )

        season = st.selectbox(
            "🌦️ Season",
            [
                "Kharif",
                "Rabi",
                "Annual"
            ]
        )

        temperature = st.slider(
            "🌡️ Average Temperature (°C)",
            5,
            45,
            25
        )

    with col2:

        ph = st.slider(
            "🧪 Soil pH",
            3.0,
            10.0,
            6.5,
            0.1
        )

        water = st.selectbox(
            "💧 Water Availability",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

    if st.button(
        "🔍 Find Suitable Crops",
        type="primary",
        use_container_width=True
    ):

        exact = recommend_crops(
            soil,
            season,
            ph,
            temperature,
            water
        )

        ranked = get_ranked_crops(
            soil,
            season,
            ph,
            temperature,
            water
        )

        if not exact.empty:

            st.success(
                f"{len(exact)} highly suitable crop(s) found."
            )

        else:

            st.warning(
                "No exact match was found. "
                "Showing crops ranked by suitability."
            )

        st.subheader("🏆 Recommended Crops")

        display_columns = [
            "Crop",
            "Season",
            "Soil",
            "Water",
            "Fertilizer",
            "Score"
        ]

        st.dataframe(
            ranked[display_columns].head(5),
            use_container_width=True,
            hide_index=True
        )

        best_crop = ranked.iloc[0]

        st.success(
            f"""
            🌱 **Best Match: {best_crop['Crop']}**

            Suitability Score: **{best_crop['Score']}/100**

            Recommended fertilizer type:
            **{best_crop['Fertilizer']}**
            """
        )


# ============================================================
# AI AGRICULTURAL ASSISTANT
# ============================================================

elif page == "🤖 AI Agricultural Assistant":

    st.header("🤖 AI Agricultural Assistant")

    st.write(
        "Ask questions about farming, crops, soil, "
        "irrigation, fertilizers, or agricultural practices."
    )

    model = st.selectbox(
        "Select Ollama Model",
        [
            "llama3.2",
            "llama3.1",
            "mistral"
        ]
    )

    question = st.text_area(
        "🌾 Ask your agricultural question",
        placeholder=(
            "Example: What crop is suitable for black soil "
            "during Kharif season?"
        ),
        height=130
    )

    number_results = st.slider(
        "Number of knowledge sources",
        1,
        5,
        3
    )

    if st.button(
        "🤖 Ask Agricultural AI",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter your question."
            )

        else:

            with st.spinner(
                "🔎 Searching agricultural knowledge..."
            ):

                documents = search_knowledge(
                    question,
                    number_results
                )

            context = "\n\n".join(
                documents
            )

            if not context:

                context = """
No uploaded agricultural documents were found.

Use general agricultural knowledge, but clearly
indicate when information should be verified locally.
"""

            with st.spinner(
                "🤖 Generating answer..."
            ):

                try:

                    answer = ask_ai(
                        question,
                        context,
                        model
                    )

                    st.subheader(
                        "🌱 AI Response"
                    )

                    st.markdown(answer)

                    if documents:

                        with st.expander(
                            "📚 View Retrieved Sources"
                        ):

                            for i, doc in enumerate(
                                documents,
                                start=1
                            ):

                                st.markdown(
                                    f"**Source {i}**"
                                )

                                st.write(doc)

                except Exception as error:

                    st.error(
                        "Could not connect to Ollama."
                    )

                    st.markdown(
                        """
                        Make sure Ollama is installed,
                        running, and that the selected
                        model has been downloaded.
                        """
                    )

                    st.code(
                        str(error)
                    )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

elif page == "📚 Agricultural Knowledge Base":

    st.header("📚 Agricultural Knowledge Base")

    st.write(
        """
        Upload agricultural PDF documents such as crop guides,
        soil manuals, government agricultural publications,
        fertilizer guides, or farming research papers.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} file(s) selected.**"
        )

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

    if st.button(
        "📥 Add PDFs to Knowledge Base",
        type="primary",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            total_chunks = 0

            progress = st.progress(0)

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                file_path = (
                    DOCUMENT_DIR /
                    uploaded_file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

                try:

                    chunks_added = (
                        add_document_to_database(
                            file_path,
                            uploaded_file.name
                        )
                    )

                    total_chunks += chunks_added

                except Exception as error:

                    st.error(
                        f"Error processing "
                        f"{uploaded_file.name}: "
                        f"{error}"
                    )

                progress.progress(
                    (index + 1) /
                    len(uploaded_files)
                )

            st.success(
                f"""
                Successfully added
                **{total_chunks} knowledge chunks**
                to the agricultural database.
                """
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "📚 Knowledge Chunks",
            collection.count()
        )

    with col2:

        st.metric(
            "📄 Stored PDFs",
            len(
                list(
                    DOCUMENT_DIR.glob("*.pdf")
                )
            )
        )

    if collection.count() > 0:

        st.subheader(
            "🔎 Test Knowledge Search"
        )

        search_question = st.text_input(
            "Search agricultural knowledge"
        )

        if search_question:

            results = search_knowledge(
                search_question,
                5
            )

            for i, result in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"Result {i}"
                ):

                    st.write(result)


# ============================================================
# IRRIGATION ADVISOR
# ============================================================

elif page == "💧 Irrigation Advisor":

    st.header("💧 Smart Irrigation Advisor")

    st.write(
        "Get basic irrigation guidance based on crop "
        "and available water."
    )

    crop = st.selectbox(
        "Select Crop",
        crop_df["Crop"].tolist()
    )

    water = st.selectbox(
        "Available Water",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    if st.button(
        "💧 Generate Irrigation Advice",
        type="primary",
        use_container_width=True
    ):

        st.markdown(
            irrigation_advice(
                crop,
                water
            )
        )


# ============================================================
# FERTILIZER ADVISOR
# ============================================================

elif page == "🧪 Fertilizer Advisor":

    st.header("🧪 Fertilizer Advisor")

    crop = st.selectbox(
        "Select Crop",
        crop_df["Crop"].tolist()
    )

    soil = st.selectbox(
        "Select Soil",
        sorted(
            crop_df["Soil"].unique()
        )
    )

    ph = st.number_input(
        "Soil pH",
        min_value=3.0,
        max_value=10.0,
        value=6.5,
        step=0.1
    )

    if st.button(
        "🧪 Generate Fertilizer Advice",
        type="primary",
        use_container_width=True
    ):

        st.markdown(
            generate_fertilizer_advice(
                crop,
                soil,
                ph
            )
        )


# ============================================================
# CROP DATABASE
# ============================================================

elif page == "📊 Crop Database":

    st.header("📊 Crop Database")

    st.write(
        "Agricultural crop information used by the "
        "recommendation engine."
    )

    st.dataframe(
        crop_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "📈 Crop Water Requirements"
    )

    water_counts = (
        crop_df["Water"]
        .value_counts()
        .reset_index()
    )

    water_counts.columns = [
        "Water Requirement",
        "Number of Crops"
    ]

    st.bar_chart(
        water_counts.set_index(
            "Water Requirement"
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 Smart Agricultural System | "
    "Streamlit + Sentence Transformers + ChromaDB + "
    "PyPDF + Ollama + Pandas"
)

