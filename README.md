# Computer Vision Project: MRI doctor

### User case
Users upload a MRI image to the UI (api) and return the answer whether the MRI image presents tumors or a healthy brain.

### Implementation
UI and API - FastAPI
build computer vision model - PyTorch framework
workflow management (modelling, training, monitoring, deploying) - metaflow
Cloud computing/storage - AWS EC2/S3


### AI experiments to run
- [ ] Test the effect of augmentation (set up experiements with or without augmentation)
- [ ] Test the effect of hyperparameters (learning_rate, batch_size)
- [ ] Test the effect of the number of filters for each conv2d layer
- [ ] Test tranfer training (VGG16)

### Features to add (ordered)
- [ ] set up metrics, optimize the model via a variety of experiments listed in the preceeding section
- [x] Generate grad-CAMs (Gradient-weighted Class Activation Mapping)
- [ ] Using new dataset (Arzheimer) to train the model
- [ ] scrape or gain programmatic access to medical imaging data [TCIA API](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides)
- [ ] Add object detection of the tumor (size)
