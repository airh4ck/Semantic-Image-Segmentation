import streamlit as st
import segmentation
from PIL import Image


def init():
    st.set_page_config(page_title="Semantic image segmentation")
    st.session_state["model"] = segmentation.create_model()
    st.session_state["feature_extractor"] = segmentation.create_feature_extractor()


def process_file(file):
    return segmentation.segment(
        Image.open(file),
        st.session_state["model"],
        st.session_state["feature_extractor"]
    )


def get_uploaded_file():
    return st.file_uploader(
        label="Choose a file",
        type=["png", "jpg", "jpeg"],
    )


def run():
    st.title("Semantic image segmentation")
    st.subheader("Upload your image and get an image with segmentation")

    file = get_uploaded_file()
    if not file:
        return

    placeholder: st.delta_generator.DeltaGenerator = st.empty()
    placeholder.info(
        "Processing..."
    )

    placeholder.empty()
    placeholder.image(process_file(file))
