# Computer Vision Project: MRI doctor

### User case
Users upload a MRI image to the UI (api) and return the answer whether the MRI image presents tumors or a healthy brain.

### Implementation
api - FastAPI
build model and load weights - torch

### Features to add
- [ ] set up metrics, optimize the model
    - [ ] break down the optimization process into sections to illustrate how to optimize a model, which can be summaried in a tech blog
    - [ ] fine-tune the hyperparameters, batch size, learning rate/different schedulers
    - [ ] build different models/use pretrained models to see what affect the final results/efficiency most. Try to use the state-of-art tools in the model optimization field.
    - [x] Generating grad-CAMs (Gradient-weighted Class Activation Mapping)
- [ ] Add object detection of the tumor (size)
- [ ] Using new dataset (Arzheimer) to train the model
- [ ] scrape or gain programmatic access to medical imaging data [TCIA API](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides)

