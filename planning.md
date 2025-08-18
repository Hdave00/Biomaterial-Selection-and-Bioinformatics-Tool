
---

# bio-project

I have a training structure in mind already but I will use the **Materials Project API** to train my RNN and build a basic interface to input values.
Can I use **TensorFlow** and **Keras** to use the Functional API to train the RNN?

Because I am not aware of how much data the **Materials Project** has that I can train my neural network on, and if I will need datasets from Kaggle like *"Bioinformatics Simulated"* for also including parameters like **molecular weight, hydrophobicity** etc. to consider how the material can be biofabricated, or even using the *"Young's Modulus of Metals"* dataset, as I’m not sure if Materials Project has those values for each material.

I’m a bit confused as I want the input to be oriented to biofabrication and also general materials and their properties but **MAINLY** I want the output to be:

* a list of materials (biological, metal, or whatever the training set has),
* their values,
* along with how the metals will interact with the target body part.

### Example

#### Input parameters:

* Area of body: Hip
* Desired young’s modulus: `"xyz GPa"`
* Desired tensile strength: `"xyz GPa"`
* Budget: 300 EUR/Unit
* Density: kg/m³ or g/cm³

#### Output parameters:

* List of appropriate materials
* Properties of matched materials:

  * Young’s modulus, tensile strength, surface charge, shear force, atomic bond,
  * reactive or non-reactive, porous or non-porous, corrosive or non-corrosive, hardness, erosion factor, glass transition temperature,
  * cytotoxicity, surface tension of the material (for gliding/friction forces)
  
* pH of blood in hip area
* Proteins found in the hip area and their fold type (using the Protein Folding Kaggle dataset), molecular weight, type of atomic bonding, toxicity, surface tension of main proteins
* Biocompatibility score
* Erosion factor of the hip bone
* Estimated lifespan of implant
* Manufacturing method
* Material forming method

---

For now those are the **ideal parameters**.

The overarching use case of this tool is to be a **calculated and well thought out compatibility check and property determiner** all in one, that also scales with usage types (modular, not just for implants).

⚠️ **Note:** This will **NOT** be a decision-making tool. It is up to the researcher or technician to make the decision based on the matched and compiled data available.

---

### Materials Project API key

```
eBp712D0xvRbeWiidnmAOVwBeuRyfq6N  -> RESTful API
```

---

## Project Structure

Web-based **Streamlit** application:

```
.
├── api
├── app
│   ├── __pycache__
│   ├── main_app.py
│   ├── model_interface.py
│   └── visualization.py
├── biomat
│   ├── bin
│   ├── etc
│   │   └── jupyter
│   ├── include
│   │   └── python3.12
│   ├── lib
│   │   └── python3.12
│   ├── lib64 -> lib
│   ├── pyvenv.cfg
│   └── share
│       └── jupyter
├── data                     -> TODO
├── models                   -> To train and test
├── requirements.txt         -> not yet done
├── templates                -> TODO
└── utils                    -> TODO
```

---

## Workflow idea (base pipeline)

1. **User selects parameters** (already done).
2. **predict\_materials:**

   * Load dataset (from `data/` folder).
   * Apply filtering (e.g., modulus within ±20%, cost < budget).
   * Rank by biocompatibility score.
3. **visualization.py:** display both raw table + plots.

---

## General Interface Outline

| Dataset Type                | Examples                                                 | Steps Required                                                                                                   |
| --------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Material Properties**     | Materials Project, Kaggle alloys, Young’s modulus data   | Normalize units (GPa/MPa), handle missing data, unify naming conventions (e.g., “Ti-6Al-4V” vs “Titanium alloy”) |
| **Biological/Protein Data** | Protein folding dataset, toxicity scores, surface charge | Group by anatomical area (“hip”), map proteins to region, extract average fold type/surface features             |
| **Manufacturing**           | High Entropy Alloys, process datasets                    | Encode as categorical, one-hot or embeddings (e.g., “EBM” -> \[0, 1, 0...])                                      |
| **Economic Factors**        | Budget, cost per unit, expected lifespan                 | Normalize cost to EUR/USD, map lifespan into years or decay score                                                |

---

## Integration Strategy

* Use **material name, composition, or body region** as keys.
* Build a **relational mapping system** (SQL joins or merged DataFrames in pandas).
* Create a final structured dataset, e.g.:

```json
{
  "target_area": "hip",
  "desired_modulus": 105,
  "desired_density": 4.5,
  "protein_fold": "alpha-helix",
  "main_material": "Ti-6Al-4V",
  "shear_strength": 800,
  "manufacturing_method": "EBM",
  "budget": 300,
  "cytotoxicity_score": 0.1
}
```

---

## Model Training Workflow

1. Preprocess each dataset independently
2. Merge into one training-ready structure
3. Feed unified data to one training pipeline (one model or modular pipeline)
4. Optionally build sub-models if needed (e.g., protein → interaction score model)

---

## Initial Checklist

1. Create core input/output schema
2. Map each dataset to parts of that schema
3. Preprocess each dataset into a consistent format
4. Merge all of the data into one training-ready dataset
5. Build a Keras Functional Model Neural Network
6. Wrap and integrate with Streamlit

---

## Streamlit Features

* **Interactive forms** → great for taking user input (target body area, modulus, budget, etc.)
* **Built-in plotting** → use matplotlib, seaborn, plotly, or altair with zero setup
* **ML model integration** → works smoothly with trained TensorFlow/Keras models
* **Live output panels** → display compatibility scores, material profiles, graphs
* **No frontend code hassle** → just Python (no HTML/CSS/JS to maintain)

---
