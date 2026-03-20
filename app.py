import os
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

# [1] API KEY (SECURE)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile"
)

# [2] VECTOR DB (PERSISTENT)
client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory="./chroma_db"
    )
)

collection = client.get_or_create_collection("career_knowledge_base")

# [3] SESSION STATE
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# [4] PDF READER
@st.cache_data
def ingest_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# [5] UI
st.title("AI Resume Analyzer")
st.markdown("Upload your resume and get AI powered career insights")

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
text = None
retrieved_context = None

# [6] INGESTION
if file:
    text = ingest_pdf(file)

    if st.session_state.uploaded_file != file.name:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )

        chunks = splitter.split_text(text)

        # store chunks
        st.session_state.all_chunks.extend(chunks)

        # add to DB
        collection.add(
            documents=chunks,
            ids=[f"{file.name}_{i}" for i in range(len(chunks))]
        )

        st.session_state.uploaded_file = file.name
        st.success("Resume ingested successfully!")

# [7] HYBRID SEARCH
def hybrid_search(query):

    # VECTOR SEARCH
    vector_results = collection.query(
        query_texts=[query],
        n_results=5
    )

    vector_docs = vector_results["documents"][0] if vector_results["documents"] else []

    # KEYWORD SEARCH
    keywords = query.lower().split()

    keyword_docs = [
        doc for doc in st.session_state.all_chunks
        if any(k in doc.lower() for k in keywords)
    ][:5]

    # MERGE
    hybrid_docs = list(set(vector_docs + keyword_docs))

    if not hybrid_docs:
        return "No relevant context found."

    # RERANK
    rerank_prompt = PromptTemplate.from_template(
        """
        Query:
        {query}
        Documents:
        {docs}
        Rank documents from most relevant to least relevant.
        """
    )

    rerank_chain = rerank_prompt | llm

    reranked_output = rerank_chain.invoke({
        "query": query,
        "docs": hybrid_docs
    })

    top_context = reranked_output.content.split("\n")[:3]
    return "\n".join(top_context)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "ATS Score",
    "Resume Analysis",
    "Skill Gap",
    "Resume Improver",
    "Career Roadmap",
    "Career Chatbot",
    "Resume Sections"
])

# FEATURE 1: ATS SCORE
with tab1:
    if st.button("Calculate ATS Score"):
        if text is None:
            st.warning("Upload resume first")
        else:
            with st.spinner("Analyzing ATS..."):
                context = hybrid_search("ATS optimization resume analysis")

                prompt = PromptTemplate.from_template(
                    """Resume:
                    {resume}
                    Context:
                    {context}
                    Evaluate ATS compatibility.
                    Return score and improvements."""
                )

                chain = prompt | llm
                result = chain.invoke({
                    "resume": text,
                    "context": context
                })

                st.write(result.content)

# FEATURE 2: ANALYSIS
with tab2:
    if st.button("Analyze Resume"):
        if text is None:
            st.warning("Upload resume first")
        else:
            with st.spinner("Analyzing Resume..."):
                context = hybrid_search("resume strengths weaknesses skill gaps")

                prompt = PromptTemplate.from_template(
                    """Context:
                    {context}
                    Provide resume feedback:
                    Strengths
                    Weaknesses
                    Skill gaps
                    Suggestions"""
                )

                chain = prompt | llm
                result = chain.invoke({"context": context})

                st.write(result.content)

# FEATURE 3: SKILL GAP
with tab3:
    role = st.selectbox("Select Target Role",
                        ["AI Engineer", "Data Scientist", "Software Developer", "Web Developer"])

    if st.button("Analyze Skill Gap"):
        if text is None:
            st.warning("Upload resume first")
        else:
            with st.spinner("Analyzing Skills..."):
                context = hybrid_search(f"skills required for {role}")

                prompt = PromptTemplate.from_template(
                    """Resume:
                    {resume}
                    Context:
                    {context}
                    Identify skill gaps for role {role}"""
                )

                chain = prompt | llm
                result = chain.invoke({
                    "resume": text,
                    "context": context,
                    "role": role
                })

                st.write(result.content)

# FEATURE 4: BULLET IMPROVER
with tab4:
    bullet = st.text_area("Paste resume bullet point")

    if st.button("Improve Bullet"):
        if bullet.strip() == "":
            st.warning("Enter a bullet point")
        else:
            with st.spinner("Improving..."):
                prompt = PromptTemplate.from_template(
                    "Rewrite professionally:\n{text}"
                )

                chain = prompt | llm
                result = chain.invoke({"text": bullet})

                st.write(result.content)

# FEATURE 5: ROADMAP
with tab5:
    if st.button("Generate Roadmap"):
        if text is None:
            st.warning("Upload resume first")
        else:
            with st.spinner("Generating roadmap..."):
                context = hybrid_search("career roadmap for resume skills")

                prompt = PromptTemplate.from_template(
                    """Resume:
                    {resume}
                    Context:
                    {context}
                    Generate career roadmap."""
                )

                chain = prompt | llm
                result = chain.invoke({
                    "resume": text,
                    "context": context
                })

                st.write(result.content)

# FEATURE 6: CHATBOT
with tab6:
    question = st.text_input("Ask career question")

    if st.button("Ask Question"):
        if question.strip() == "":
            st.warning("Enter a question")
        else:
            with st.spinner("Thinking..."):
                context = hybrid_search(question)

                prompt = PromptTemplate.from_template(
                    """Context:
                    {context}
                    Question:
                    {question}"""
                )

                chain = prompt | llm
                result = chain.invoke({
                    "context": context,
                    "question": question
                })

                st.write(result.content)

# FEATURE 7: SECTION EXTRACTOR
with tab7:
    if st.button("Extract Resume Sections"):
        if text is None:
            st.warning("Upload resume first")
        else:
            with st.spinner("Extracting..."):
                prompt = PromptTemplate.from_template(
                    """Extract:
                    Name
                    Education
                    Skills
                    Projects
                    Experience
                    Certifications
                    Resume:
                    {resume}"""
                )

                chain = prompt | llm
                result = chain.invoke({"resume": text})

                st.write(result.content)

# FEATURE 8: JD MATCHING
st.divider()
st.header("Job Description Matching")

jd_file = st.file_uploader("Upload Job Description", type=["pdf"], key="jd")

if jd_file:
    jd_text = ingest_pdf(jd_file)

    if st.button("Job Match Analysis"):
        if text is None:
            st.warning("Upload resume first")
        else:
            with st.spinner("Matching..."):
                context = hybrid_search("resume job description match")

                prompt = PromptTemplate.from_template(
                    """Resume:
                    {resume}
                    Job Description:
                    {jd}
                    Context:
                    {context}
                    Compare and give match score."""
                )

                chain = prompt | llm
                result = chain.invoke({
                    "resume": text,
                    "jd": jd_text,
                    "context": context
                })

                st.write(result.content)