import streamlit as st
import sys
import os

st.title("EcoGuard ML - Test App")

try:
    # Test imports
    st.write("Testing imports...")
    from models.ml_predictor import EcoMLPredictor
    st.write("✅ ML predictor imported successfully")
    
    # Test initialization  
    st.write("Testing ML predictor initialization...")
    predictor = EcoMLPredictor()
    st.write("✅ ML predictor initialized successfully")
    
    # Test data generation
    if st.button("Test Data Generation"):
        with st.spinner("Generating test data..."):
            data = predictor.generate_sample_data(50)
            st.write(f"✅ Generated {len(data)} records")
            st.write("Data preview:")
            st.dataframe(data.head())
    
    # Test model training
    if st.button("Test Model Training"):
        with st.spinner("Training models..."):
            data = predictor.generate_sample_data(100)
            pop_model, pop_score = predictor.train_population_model(data)
            risk_model, risk_accuracy = predictor.train_risk_model(data)
            
            st.write(f"✅ Population model R² score: {pop_score:.4f}")
            st.write(f"✅ Risk model accuracy: {risk_accuracy:.4f}")
    
    st.success("All tests completed successfully!")
    
except Exception as e:
    st.error(f"Error: {e}")
    st.write("Error details:")
    import traceback
    st.code(traceback.format_exc())
