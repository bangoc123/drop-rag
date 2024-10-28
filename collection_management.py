import streamlit as st
import pandas as pd

# Define dialog function
@st.dialog("Popup Dialog")
def open_dialog(message, func, arg):
    st.write(message)
    if st.button("Confirm"):
        func(arg)
        st.session_state.open_dialog = None
        st.rerun()

    if st.button("Cancel"):
        st.session_state.open_dialog = None
        st.rerun()


@st.dialog("Collection List", width="large")
def list_collection(session_state, load_func, delete_func):
    if "client" in session_state and session_state.client:
        collections = session_state.client.list_collections()

        # Prepare data for display in a DataFrame
        collection_data = [
            {
                "Collection Name": collection.name,
                "Metadata": str(collection.metadata)
            }
            for collection in collections
        ]

        # Convert to DataFrame
        df = pd.DataFrame(collection_data, columns=["Collection Name", "Metadata", "Action"])


        head_col1, head_col2, head_col3 = st.columns([2, 2, 2])
        with head_col1:
            st.write("**Collection Name**")
        with head_col2:
            st.write("**Metadata**")
        with head_col3:
            st.write("**Action**")
        st.markdown("---")

        for index, row in df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.write(row["Collection Name"])
                if st.session_state.random_collection_name == row["Collection Name"]:
                    st.markdown("""
                        <span style="color: red;">This collection is currently in use.</span>
                    """, unsafe_allow_html=True)
            with col2:
                st.write(row["Metadata"])
            with col3:
                # Load button (you can modify as needed)
                if st.button("Load", key=f"load_{index}"):
                    load_func(row["Collection Name"])
                    # Your load action here, such as reloading the page or setting a variable
                    st.rerun()
            with col4:
                if st.button("Delete", key=f"delete_{index}"):
                    delete_func(row["Collection Name"])
                    st.rerun()
            st.markdown("---")  
