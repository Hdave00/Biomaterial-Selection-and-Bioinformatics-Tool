https://www.kaggle.com/datasets/marquis03/metal-organic-frame-materials-prediction

Metal-Organic Frameworks (MOF) are a class of crystalline materials connected by coordination bonds between metal ions (or metal clusters) and organic ligands. MOF materials have porous structure, highly tunable and huge specific surface area, which makes them have wide application potential in fields such as adsorption, gas storage, separation, catalysis and so on. Prediction synthesis refers to the prediction and design of synthetic routes and conditions of new MOF materials through computer simulation and machine learning methods.

However for my use case this dataset will be a baseline of metal porosity and extremely useful for learning about drug delivery scaffolds, coatings, and advanced biomaterials. This will be possible because the **RAC feature data combines the pore geometry of the MOF with the chemical composition (such as metal nodes, ligands, and functional groups) to obtain the feature vector** and the dataset is already split into training and testing csv datasets with the following columns:

ASA [m^2/cm^3] – CH4HPSTP	Float and Int, RAC eigenvector of sample MOF
temperature	                Float, material synthesis temperature
time –                      Float, material synthesis time
solvent1 –                  solvent3	Int, organic solvent used in material synthesis
additive	                Int, additives used in the synthesis of materials
param1 –                    param5	Float, organic solvent-related properties (normalized)
additive_category	        Int, additive category