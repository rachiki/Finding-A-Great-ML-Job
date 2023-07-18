Our project about abstractive summarization focused on using a discriminator model as 
1. a metric for summary quality and 
2. as a part of a GAN pipeline.

This submission contains the presentations we did on the topic, a report describing our work in more detail 
and the "Code" section, containing all necessary parts for training the discriminator model and generator models 
and some of the trained discriminator models.
Generators were not saved as the generator training step of our GAN approach did not stabilize 
and the resulting generators were not worth making saves for.

The best way to work with the code is to upload the "Abstractive summarization.ipynb" into google colab and the NLP_project folder on drive. 
The ability to collapse sections on colab makes the code much clearer and easier to handle. 
You can load either pretrained or checkpoints of models we trained there and either do inference with them or continue training. 
This can be done for different datasets and models.
A guide of how exactly it is used is included in the beginning of the notebook.

This the more polished and essential code we have, but other parts of our work are included in the repository.

Note: We wanted to include the most recent version of all finetuned discriminator models, 
but we could not get the big files containing the weights for the distilbart models to upload on GitLab without crashing, 
to access and add them you can use this link https://drive.google.com/drive/folders/1YSVcy-oSniGBd4wfzOJgyZepaQ9JwRxl?usp=sharing.
