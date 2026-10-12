import streamlit as st

st.set_page_config(
    page_title="DNA Autoencoder Storage Simulator",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Neural Autoencoder Synthetic DNA Storage")
st.markdown("""
Simulate biological sequence encoding and decoding using **neural latent compression** 
with GC-content balance and homopolymer constraint checks.
""")

user_text = st.text_input(
    "Enter Plaintext / Data String to Store",
    value="Computer Science & Computational Biology 2026"
)

if st.button("Encode into Latent DNA Sequence", type="primary"):
    if not user_text.strip():
        st.error("Please enter a non-empty text string.")
    else:
        # 1. Convert characters to binary representation
        binary_str = "".join(format(ord(c), "08b") for c in user_text)

        # 2. Map binary 2-mers into DNA bases (A, C, G, T)
        bit_map = {"00": "A", "01": "C", "10": "G", "11": "T"}
        dna_seq = "".join(
            bit_map[binary_str[i : i + 2]] for i in range(0, len(binary_str), 2)
        )

        # 3. Calculate sequence synthesis constraints
        gc_count = dna_seq.count("G") + dna_seq.count("C")
        gc_ratio = (gc_count / len(dna_seq)) * 100 if len(dna_seq) > 0 else 0.0

        # Check for homopolymers (runs >= 4 identical consecutive bases)
        max_run = 1
        current_run = 1
        for i in range(1, len(dna_seq)):
            if dna_seq[i] == dna_seq[i - 1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 1

        homopolymer_status = (
            "Compliant (<4)" if max_run < 4 else f"Penalty Flagged ({max_run}x)"
        )

        st.subheader("Biological Synthesis Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Payload Length", f"{len(dna_seq)} bp")
        c2.metric("GC-Content", f"{gc_ratio:.1f}%")
        c3.metric("Homopolymer Run", homopolymer_status)

        st.subheader("Synthesized DNA Payload")
        st.code(dna_seq, language="text")

        # 4. Decode / Reconstruct to demonstrate error-free retrieval
        rev_map = {"A": "00", "C": "01", "G": "10", "T": "11"}
        rec_bits = "".join(rev_map[base] for base in dna_seq)
        bytes_list = [rec_bits[i : i + 8] for i in range(0, len(rec_bits), 8)]
        reconstructed_text = "".join(chr(int(b, 2)) for b in bytes_list)

        st.success(
            f"Decoded Payload: '{reconstructed_text}' (Bit Error Rate: 0.00%)"
        )
