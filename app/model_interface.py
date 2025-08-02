import pandas as pd
import numpy as np

# TODO
def predict_materials(user_input):
    
    """ Takes user inpit parameters and predicts materials to be shown. """

    # Load actual model and preprocess here in future, for now return dummy results
    results = pd.DataFrame({
        "Material": ["Ti-6Al-4V", "CoCrMo", "UHMWPE"],
        "Young’s Modulus (GPa)": [110, 230, 0.9],
        "Tensile Strength (MPa)": [950, 1000, 45],
        "Density (g/cm³)": [4.43, 8.29, 0.93],
        "Biocompatibility Score": [0.93, 0.88, 0.95],
        "Estimated Lifespan (yrs)": [20, 15, 10],
        "Manufacturing Method": ["EBM", "Casting", "Compression Molding"]
    })
    return results