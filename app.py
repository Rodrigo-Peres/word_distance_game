# -*- coding: utf-8 -*-
"""

Created on Apr 2022

@author: Rodrigo Peres

"""

import fasttext
import numpy as np
import pandas as pd
import streamlit as st

from scipy import spatial

st.write(
    """
# Word Distance App

This app predicts the distance between a user-entered word and a predefined word.

"""
)

ft_en = fasttext.load_model("cc.en.300.bin")


def word_distance(predefined_word, user_word):

    word_vector_1 = ft_en.get_word_vector(predefined_word.strip().lower())
    word_vector_2 = ft_en.get_word_vector(user_word.strip().lower())

    return spatial.distance.cosine(word_vector_1, word_vector_2)


def main():
    st.title("Word Distance")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Word Distance App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    predefined_word = "queen"
    user_word = st.text_input("Word", "Type Here")
    result = ""
    if st.button("Predict"):
        result = word_distance(predefined_word, user_word)
    st.success("The distance is {}".format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")


if __name__ == "__main__":
    main()
