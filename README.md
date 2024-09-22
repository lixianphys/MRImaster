# MRIMaster: AI-supported medical imaging classifier

### User case
Users upload a MRI image to the UI (api) and return the answer whether the MRI image presents tumors or a healthy brain.

### Implementation
UI and API - FastAPI
build computer vision model - PyTorch framework
workflow management (modelling, training, monitoring, deploying) - metaflow
Cloud computing/storage - AWS EC2/S3

```
/root
├── app.py
├── templates
│   └── result.html
├── static
    ├── styles.css
    └── uploads
        └── cf15ffea09884dc6b9aa6d7b293c31c2.png
```



### AI experiments to run
- [ ] Test the effect of augmentation (set up experiements with or without augmentation)
- [ ] Test the effect of hyperparameters (learning_rate, batch_size)
- [ ] Test the effect of the number of filters for each conv2d layer
- [ ] Test tranfer training (VGG16)

### Features to add (ordered)
- [x] Generate grad-CAM (Gradient-weighted Class Activation Mapping)
- [ ] Using new dataset (Arzheimer) to train the model
- [ ] scrape or gain programmatic access to medical imaging data [TCIA API](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides)
- [ ] Add object detection of the tumor (size)
