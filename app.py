import streamlit as st
import torch
import pickle

from model import MiniGPT

# -----------------------------
# Load Vocabulary
# -----------------------------
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

with open("idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

vocab_size = len(word_to_idx)

# -----------------------------
# Load Model
# -----------------------------
model = MiniGPT(vocab_size=vocab_size)

model.load_state_dict(
    torch.load(
        "mini_gpt_model.pth",
        map_location="cpu"
    )
)

model.eval()

# -----------------------------
# Text Generation Function
# -----------------------------
def generate_text(prompt, max_new_tokens=20):

    words = prompt.lower().split()

    ids = [
        word_to_idx.get(
            word,
            word_to_idx["<UNK>"]
        )
        for word in words
    ]

    if len(ids) == 0:
        return "Please enter a prompt."

    for _ in range(max_new_tokens):

        input_ids = torch.tensor(
            [ids],
            dtype=torch.long
        )

        with torch.no_grad():
            outputs = model(input_ids)

        next_token_id = torch.argmax(
            outputs[0, -1]
        ).item()

        ids.append(next_token_id)

    generated_text = " ".join(
        idx_to_word.get(i, "<UNK>")
        for i in ids
    )

    return generated_text


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Mini GPT",
    page_icon="🤖"
)

st.title("🤖 Mini GPT Text Generator")

prompt = st.text_input(
    "Enter Prompt",
    placeholder="e.g. artificial intelligence"
)

if st.button("Generate"):

    with st.spinner("Generating text..."):

        output = generate_text(
            prompt,
            max_new_tokens=20
        )

    st.subheader("Generated Text")
    st.write(output)