import streamlit as st
from PIL import Image
import re
from io import BytesIO

import segmentation


def init():
    st.set_page_config(page_title="Semantic image segmentation")
    st.session_state["model"] = segmentation.create_model()
    st.session_state["feature_extractor"] = segmentation.create_feature_extractor()


@st.experimental_memo(show_spinner=False)
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


def download_button(file, name, format):
    st.download_button(
        label="Download processed image",
        data=file,
        file_name=name,
        mime="image/" + format
    )


def run():
    st.title("Semantic image segmentation")
    st.subheader("Upload your image and get an image with segmentation")

    file = get_uploaded_file()
    if not file:
        return

    placeholder = st.empty()
    placeholder.info(
        "Processing..."
    )

    image = process_file(file)
    placeholder.empty()
    placeholder.image(image)

    filename = file.name
    format = re.findall("\..*$", filename)[0][1:]

    image = Image.fromarray(image)

    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_image = buf.getvalue()

    download_button(byte_image, filename, format)
