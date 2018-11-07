# peruvian-food-identifier

First of all thanks to Jeremy Howard and Rachel Thomas for such great course as Deep Learning for coders v3!

This small project consists in build a deep learning model to classify 10 types of peruvian food using the fastai library:
* Ceviche
* Ají de gallina
* Lomo saltado
* Causa limeña
* Rocoto relleno
* Tacu tacu
* Anticuchos
* Pan con chicharrón
* Olluquito
* Picante de cuy

# Step 1: Get the dataset

The dataset can be obtained through [Google images](https://images.google.com/), searching for each category and then using a Javascript code in the browser to get a text file which contains all the urls. 

So, first press the <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>J</kbd> in Windows and then type the following code:
```javascript
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```

# Step 2: Download the images

After obtain all the urls files, the next step is download the images. In order to do that, I will use the fastai function `download_images` as follows: 
```python
download_images(path/file, dest, max_pics=400)
```

# Step 3: Train the model

Once the images are stored in the cloud environment (in my case GCP), the next step is create the Data object and the Learner object:

```python
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=error_rate) 
```
