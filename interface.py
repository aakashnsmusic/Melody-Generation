import streamlit as st
import melody_generator 
st.header("Melody Generator")
seed=st.text_input("Enter seed")
temp=st.number_input("Enter temperature")
fn=st.text_input("Enter filename")
ss=st.number_input("Enter number of bars to be generated",step=1)
but=st.button("Enter")
ss=int(16*ss)
if but:
    mg= melody_generator.MelodyGenerator()
    melody=mg.generate_melody(seed,ss,64,temp)
    mg.save_melody(melody,file_name=fn)