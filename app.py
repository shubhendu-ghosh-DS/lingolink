import streamlit as st
import requests
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

st.markdown('<p style="text-align:center; font-size: 70px; margin-bottom: 50px; margin-top: 0px;"><b>LingoLink</b></p>', unsafe_allow_html=True)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def translate(text, src_lang, tgt_lang):
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    output = translator(text, max_length=400)
    return output[0]["translation_text"]

lang_dict = {
    "Assamese":"asm_Beng",
    "Bengali":"ben_Beng",
    "Chinese":"zho_Hans",
    "Czech":"ces_Latn",
    "Danish":"dan_Latn",
    "English":"eng_Latn",
    "German":"deu_Latn",
    "Greek":"ell_Grek",
    "Finnish":"fin_Latn",
    "French":"fra_Latn",
    "Gujarati":"guj_Gujr",
    "Hindi":"hin_Deva",
    "Japanese":"jpn_Jpan",
    "Kannada":"kan_Knda",
    "Korean":"kor_Hang",
    "Maithili":"mai_Deva",
    "Malayalam":"mal_Mlym",
    "Odia":"ory_Orya",
    "Portuguese":"por_Latn",
    "Russian":"rus_Cyrl",
    "Sanskrit":"san_Deva",
    "Tamil":"tam_Taml",
    "Telugu":"tel_Telu",
    "Urdu":"urd_Arab",
}
languages = tuple(lang_dict.keys())


def main():
    cols=st.columns(2)
    with cols[0]:
        src_lang = st.selectbox('From', languages)
    with cols[1]:
        tgt_lang = st.selectbox('To', languages)

    src_lang = lang_dict[src_lang]
    tgt_lang = lang_dict[tgt_lang]

    text = st.text_area("Enter your text below:")
    
    
    if st.button("Translate"):
        if src_lang and tgt_lang and text:
            result = translate(text, src_lang, tgt_lang)
            st.text_area("", result)
        else:
            st.warning("Please provide source language, target language, and text to translate.")

if __name__ == "__main__":
    main()
