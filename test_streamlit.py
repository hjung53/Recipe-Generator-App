import streamlit as st

st.title("✅ Streamlit is working!")
st.write("All packages are installed correctly.")
st.write("Now you just need to:")
st.write("1. Download the fridge_recipe_app.py file")
st.write("2. Navigate to the folder where you saved it")
st.write("3. Run: streamlit run fridge_recipe_app.py")

st.success("Your environment is ready! 🎉")

if st.button("Test Button"):
    st.balloons()
    st.write("Everything works!")
