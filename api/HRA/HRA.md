The HRA's API will be called for:

1. Context modelling, so instead of just predicting "x material has y young's modulus" i could say, "X alloy matches cortical bone better than trabecular bone in femur". HRA also might be able to give us tissue-level details of human anatomy for reference during biomaterial selection and production, so that includes biomechanical and biochemical environments.

2. HRA also lets me do the UI/UX how I want ie, I can integrate 3D anatomical context for implants -> hip, knee, spine, dental etc.
But I would have to integrate it with Streamlit to highlight **where** in the body a material is being considered, to perhaps make the tool more intuitive.

3. Condition modelling could also be a big plus, as HRA provides microenvironment data (cell types, tissues, sometimes biomechanical parameters). So I could theoretically, model pH variation, oxygen levels, mechanical loading in different body sites, then test how the materials perform under those conditions. This could also be useful for the testing phase, comparing the output of the RNN and also using HRA as sort of a fallback for edge cases.

4. If in the future i want to have an exact fit, i could enhance the output but showing graphically, the anatomical site, "material fit score" with a visual overlay using HRA. -> mainly for scaling, i doubt this is a core feature. 

so we basically want to pull: Anatomy Structures, The conditions in the tissue or the biomechanics etc, and the Cytotoxity within scope. Crosslink that with the datasets to create a realtime guide of the biomaterial selection and fabrication, listing clearly, how the material should be formed and what comes with fabricating a certain implant/prosthetic a certain way, and how that contradicts to the part of the human body we are considering (visualize using HRA), to ultimately get a compatibility score. 