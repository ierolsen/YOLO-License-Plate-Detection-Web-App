# LISENCE PLATE DETECTION USING YOLO


![plate_detector](https://user-images.githubusercontent.com/30235603/201517094-cda6ed67-2cd7-4796-8034-982d86354ee5.gif)

![85](https://user-images.githubusercontent.com/30235603/202149654-6a0d8941-e88e-4ef0-9530-7194778e8ff5.png)


---


## Installations:

### 1- Create an Environment:
#### For Windows:

```bash
conda create -n plate_detector python=3.9
activate plate_detector
```

### 2-Install Libraries

```bash
pip install -r requirements.txt
```

---

## Project Architecture

    Labeling -> Training -> Save Model -> OCR and Pipeline -> RESTful API with Flask

Train Images -> Labeling -> Data Preprocessing -> Train Deep Learning Model -> Model Artifacts -> Make Object Detection -> OCR - Text Extract -> Model Pipeline -> Model Artifacts -> RESTful API with Flask

---


## Image Annotation Tool

https://github.com/heartexlabs/labelImg

Download this repo and follow the instructions,

![1](https://user-images.githubusercontent.com/30235603/202146702-73727327-5d00-4f89-a567-ffcd1a3bc417.png)

By following the instructions, it is way easy to install.

![2](https://user-images.githubusercontent.com/30235603/202146703-e71fc4c7-eb90-4a33-9ce7-15f361bb0cd0.png)


---

## Label Image

After opening the tool, click **Open Dir** and choose the directory where the dataset are, then the tool will automatically load the images.
When you get the image in the GUI, click **Create RectBox** and draw it then **Save** the **xml** files into the dataset.

![3](https://user-images.githubusercontent.com/30235603/202146726-035e32ee-7549-42e1-ad84-581069b924e4.png)


---



## **Section 1 - XML to CSV**
### **Notebook: 01_xml_to_csv.ipynb**

In this section, I will parse the label information and save them as **csv** .


![4](https://user-images.githubusercontent.com/30235603/202146765-787f5531-486d-4cb9-bead-03b419e9b547.png)

![5](https://user-images.githubusercontent.com/30235603/202146768-9dc2d9bd-9d56-4e85-b617-4d32029a2206.png)


I will apply all these steps for all images.

![6](https://user-images.githubusercontent.com/30235603/202146774-d53d0650-a0d3-4668-8d15-2a677dbc207a.png)


---


## **Section 2 - Data Processing**
### **Notebook: 02_Object_Detection.ipynb**

#### 2.1- Load and Get Image Filename

In this section, firstly, I will get the images path in order to read images.
In the last section, I only saved path of xml files, that is not enough to read images using OpenCV.

![7](https://user-images.githubusercontent.com/30235603/202146776-17496087-de82-47c9-9bed-799a2a135e40.png)


As you can see in the **line6**, I am able to get the path of one of the images.


![8](https://user-images.githubusercontent.com/30235603/202146779-c5c5043b-5e8a-41b3-8a76-98d6772d480b.png)

#### 2.2- Verify Labeled Data

In this part, what I will do is that I will try to draw the bounding box in order to be sure if the values are correct.


I am starting this step by reading first image of the image list. And then, I will draw the bounding box. Here is the result.


![9](https://user-images.githubusercontent.com/30235603/202146780-375aaf5b-7973-4888-aad9-2787d23cabf1.png)


#### 2.3- Data Preprocessing

Here, I will load the images and convert them into array using Keras. Also I will normalize the data which are labels and images.


![10](https://user-images.githubusercontent.com/30235603/202146783-eef44d7c-7d5f-4b3f-830f-d87ff1e832c4.png)


As you can see, bounding box coordinates are normalized. I will apply these steps to all coordinates and at additional, I will normalize the images.


![11](https://user-images.githubusercontent.com/30235603/202146787-affe2b8d-0c74-47d2-8754-9b06aa64ea83.png)



#### 2.4- Train Test Split

In this section, I will split the data as **train** and **test** by splitting 80% train size. Before splitting the data, values must be **np.array**.


![12](https://user-images.githubusercontent.com/30235603/202146789-cc76e602-37f3-44fa-965b-e891218c7a6f.png)


#### 2.5- Deep Learning Model

In this section, I will train a model for prediction. But I am not going to train a model from scratch, instead I will use **Transfer Learning** or as known as **pre-trained models** which are **MobileNetV2**,  **InceptionV3**,  **InceptionResNetV2**.

I am starting with import all necessary libraries that I will be using.


![13](https://user-images.githubusercontent.com/30235603/202146793-9b97f6e9-1e5a-4a08-92cc-41ba697c3c93.png)


#### 2.6- Building Neural Network

![14](https://user-images.githubusercontent.com/30235603/202146794-0c3ee4ab-972c-40e1-bf1c-a55aac4064e4.png)


#### Some Explanations:

-  **inception_resnet.trainable = False** means, I will use the same **weights** for this project.

-  The reason why last Dense layer is 4 is, it is our number of labels.



#### 2.7- Compiling the Model


![15](https://user-images.githubusercontent.com/30235603/202146797-8cb2cdfe-75ea-4b74-bd35-c2515e5c2e84.png)


#### 2.8- Training

![16](https://user-images.githubusercontent.com/30235603/202146799-bda8b969-eaab-4d71-830a-efecaf71b548.png)


#### 2.9- Opening TensorBoard Log File

In order to check log file, type this to the terminal

```bash
tensorboard --logdir="logs"
```

It will direct us to the localhost.


![17](https://user-images.githubusercontent.com/30235603/202146803-a5d79502-0163-483f-96d2-196536b91325.png)

---

## **Section 3 - Pipeline for Object Detection**
### **Notebook: 03_Make_Prediction.ipynb**

In this section, I will load the model and create a prediction pipeline, and also I will see that how the model draw the bounding boxes.

As first step, I am starting with loading the model.

#### 3.1- Load Model

![18](https://user-images.githubusercontent.com/30235603/202146827-6176b9cb-e272-4916-8030-6da03695ee8c.png)


For testing the model, I will randomly open an image and it will be tested by the model. The idea is that the model will predict and draw the coordinates of the plate. Basically, the model will return the bounding box

#### 3.2- Testing the Model

In this part, I will test the model in 2 different ways one of which are by using original images that has its own original sizes, other way is that I will use reshaping image which has 224,224. I will check the results how they will affect.


![19](https://user-images.githubusercontent.com/30235603/202146829-b35066ae-4fc8-41bd-bab5-77bb39ffdf4f.png)

![20](https://user-images.githubusercontent.com/30235603/202146832-11a5395d-9105-4553-815e-fb60acc958af.png)


#### 3.2.1- Prediction

![21](https://user-images.githubusercontent.com/30235603/202146841-571f0212-4f6c-4596-82ba-518d0aff09b1.png)


There it is, I got the prediction of the coordinates but these coordinates are normalized, I must convert back the results from normalization.

#### 3.2.2- Denormalization

From the Section-2, 


![22](https://user-images.githubusercontent.com/30235603/202146843-a1445b96-4d96-4c8f-99f2-814963a0142c.png)


I made a normalization above the image, it is so simple to denormalization the image again. For this, I will multiple with the original height and width. 

That’s all!

![23](https://user-images.githubusercontent.com/30235603/202146846-d68ffc89-a0a8-444c-9400-949d098b0ff8.png)


Now, I got the denormalized coordinates. 

#### 3.3- Draw Bounding Box


![24](https://user-images.githubusercontent.com/30235603/202146848-4d06e0b6-edee-48ba-8aba-22ea79e44d10.png)


The model predicts it really well for this image.
In order to get better result, I have to feed to Neural Network with a lot of data. 
#### 3.4- Pipeline


![25](https://user-images.githubusercontent.com/30235603/202146853-7eed7bd4-059d-489d-ab7b-ca89016ae473.png)

![26](https://user-images.githubusercontent.com/30235603/202146855-666bb02f-dec0-4acc-a52e-0b15ea5b7645.png)


As it is seen, the model can not predict well.

#### **NOTE**: I have retrained my model again, and nothing changed. The only way to solve it is feeding the model with more data.

---


## **Character Recognition - OCR (Optical Character Recognition)**

In this section, I will crop the plate and read their characters using *PyTesseract**. But when I do this, I will use the proper image which is the first one, because the model only works well on this image :))


![27](https://user-images.githubusercontent.com/30235603/202146860-5f62d4f3-6d83-453d-b6b5-4f2656c63432.png)



#### 3.5- Crop Bounding Box

![28](https://user-images.githubusercontent.com/30235603/202146867-46f4dd7b-3d96-49ca-8da4-cca9d8e5e993.png)


#### 3.6- Extract Text from the Plate


![29](https://user-images.githubusercontent.com/30235603/202146870-0eb61555-2753-4c75-a8f1-2c53ab59af77.png)


As you can see, **PyTesseract** can not extract the text well too, it is because of the angle of the plate. In this kind of situations, PyTesseract can not work properly. In the next part, I may be fixing it.


---

## **Section 4 - Web App Using Flask**
### **app. py**

In this section, I will be developing a web app where we can upload car plates for detecting and reading them. For this, I will use **Flask**.

>>> In order to make it clear, I am not planing to explain what I am doing with HTML or etc. I will only explain the python side of Flask.


#### 4.1- Creating First Flask App

In order to test the installation of **Flask**, I will create a quick app which says “Hello World”.

![30](https://user-images.githubusercontent.com/30235603/202146874-11f88e89-bb55-4f40-8f9a-ade2c3e541c6.png)


Then, by typing **python app.py** to the **cmd**, it will return a localhost where we can see out outputs.

And here is the result:

![31](https://user-images.githubusercontent.com/30235603/202146877-801ebc5c-0082-4273-9f76-cda113cb6e43.png)


As can be seen, everything looks cool.

#### 4.2- Bootstrap Installation

In order to import **Bootstrap** into your project, you should firstly create a directory called **templates**. after this step, you should also create one **html** file called **layouts.html** or you can name it **base.html** it is up to you..

After completing these steps, go to official website of **Bootstrap** which is https://getbootstrap.com/docs/5.2/getting-started/download/ and copy the CDN links of CSS and JS.

![32](https://user-images.githubusercontent.com/30235603/202146879-5fd44ba3-88ec-459c-b6db-da53c37e712b.png)


Then, paste them into **head tags** of **layouts.html** file you have created before.


![33](https://user-images.githubusercontent.com/30235603/202146886-81bc29da-9c98-4806-9dd7-66dd96782987.png)



#### 4.3- NAVBAR


![34](https://user-images.githubusercontent.com/30235603/202146890-1e9dd1e6-50c5-42ea-bc4e-1e68e446b21e.png)


I have had to design the **navbar** inside of **layout.html** file, because navbar will be contained whole pages of this project. It is one of the base things.


![35](https://user-images.githubusercontent.com/30235603/202146893-d7997187-2c2a-4d6f-9112-22fd63ef4a8d.png)


#### 4.4- FOOTER


![36](https://user-images.githubusercontent.com/30235603/202146896-8d484980-09b0-4cf3-b582-b79c65100977.png)



#### 4.5- INDEX File

Index file is a page which it will be my main page where I can load the images etc. I will inheritance the **layout.html** file for getting **NAVBAR** and **FOOTER** and also **Bootstrap**.

For this I will add
    {% block body %}
        
    {% endblock %}
        
Inside of the **layout.html** and I will create a new html file called **index.html** as I said before, it will be my main page where I will upload the images and get the results.


![37](https://user-images.githubusercontent.com/30235603/202146898-28939742-2919-44f9-b6ea-55eba978e466.png)


Here is the result 


![38](https://user-images.githubusercontent.com/30235603/202146899-6dc03610-6767-4462-b7d9-96bf774d7b36.png)



#### 4.6- UPLOAD FORM 


![39](https://user-images.githubusercontent.com/30235603/202146901-9cf66945-b0e8-4ae5-9451-b5ce501f4213.png)

![40](https://user-images.githubusercontent.com/30235603/202146902-0ade06b8-e7fb-4916-83a2-a7d87ccd0df1.png)




#### 4.7- FILE UPLOAD

In this section, I will code the part which will be running when **Upload Button** is clicked.

I am expecting a data which is an image as POST form. I need to receive the data and save that into a folder that is called **static**.

For that I will create one more folder called **static** and it will also have one more folder called **upload**.

**Most important** thing here is, all these files have to be created inside of working directory where **Flask App** is.


![41](https://user-images.githubusercontent.com/30235603/202146905-65f89ac3-ef80-4b17-9f66-f56729cadaf9.png)


First I started with defining BASE_PATH which is current working directory and then defined UPLOAD_PATH where images are saved.

When an images are uploaded, it is **POST** method, so inside of **if statement** I checked this then I got the name of the image which is **image_name** where I defined inside of **UPLOAD FORM** in **index.html** file. If you check the third image above, you will see the **name=image_name**.

Then I got the **filename** of the image for the next step which is **saving**, and as last step, I defined **save** method.

And, when I try to load an image, here is what happens in the **static/upload** file.


![42](https://user-images.githubusercontent.com/30235603/202146907-29f37d2e-ccd3-4927-ae85-6396caddb4ef.png)



#### 4.8- EDITING THE DEEP LEARNING MODEL

In this section, using the uploaded images, I will make prediction using trained deep learning model which I have done before.

For this process, I need this function,


![25](https://user-images.githubusercontent.com/30235603/202146853-7eed7bd4-059d-489d-ab7b-ca89016ae473.png)

![43](https://user-images.githubusercontent.com/30235603/202146912-80f20b61-c0d1-4c16-8675-c2ca11f38537.png)


I have changed these covered parts, let me explain why.

The function will get two parameters, first one i **path** other one is **filename**, I will upload more than one images, so the model will predict them, in order to save them properly it is necessary. Other reason is that I will also save drawn rectangle images. That is also why.

![44](https://user-images.githubusercontent.com/30235603/202146914-c4c56693-aa22-44f4-88e5-619b0b1a6e7c.png)


I have also defined one more function for **OCR**. This function will be used in OCR process as you can guess. This function will also get two parameters like in last function, I will also save the **Cropped license Plate** into **static/roi** folder which I have created before.

#### 4.9- IMPLEMENTING THE DEEP LEARNING MODEL TO THE FLASK APP

First, I import my **OCR function**: from deeplearning import OCR

I run the **app. py** and upload a image to see the output:

![45](https://user-images.githubusercontent.com/30235603/202146921-c68ad173-d7e2-4f57-9bd4-37dd033b221d.png)


As can be seen, the model predict it ( as you may remember in previous sections, the model can not work well because of that fed with less data.) but **my main idea was that learning how to implement a Deep Learning model to Flask App.** so this project will be so informative for me. 

![46](https://user-images.githubusercontent.com/30235603/202146926-011dfe4a-dca0-49cb-852a-a17a1000536a.png)


I also want you to realize these covered parts of the image, the drawn bounding box, cropped plate and uploaded images are saved into these folders. At least, the algorithm works well! :) I wish the model too..

#### 4.10- DISPLAY OUTPUTS in HTML PAGE

In this section, I will display these images which are original image and drawn bounding box image.

For this, inside of **render_templates** I will add a parameter which is **upload**. If it is **True**, it will display the results.

![47](https://user-images.githubusercontent.com/30235603/202146930-8caf6c0e-b528-41a5-87c4-adf0ba20aaee.png)


Here I specified the variables in order to use in the **index.html** file.


![48](https://user-images.githubusercontent.com/30235603/202146931-bc4e757b-9f13-4081-9eba-9279c3e2f6a1.png)


In this **index.html** page, I created some tables for putting the images.

![49](https://user-images.githubusercontent.com/30235603/202146652-ffe9baa1-48ba-4b7a-a213-2980d19caad3.png)


And the we are able to see the prediction! I will also add **Cropped License Plate** and its text version.

![50](https://user-images.githubusercontent.com/30235603/202146661-1acfe2bc-570e-4c7b-b263-5aab8cb19c75.png)


I have also decided to try with **YOLO**. 

Let’s see..

---

## **Section 5 - License Plate Detection with YOLOv5**
### **Notebook: 05_YOLO.ipynb**

The biggest problem of this project is the accuracy of the model. The model has really low precision in detecting the license plates. In order to solve this problem, I will use YOLOv5 which is most powerful object detection model.

One of the biggest differences in YOLO is bounding box format. 
X and Y positions are preferred to the **center of the bounding box**.

![51](https://user-images.githubusercontent.com/30235603/202146665-199b54d2-8ecd-4454-a6ca-f552da40aa9f.png)
Source: https://haobin-tan.netlify.app/ai/computer-vision/object-detection/coco-json-to-yolo-txt/


![52](https://user-images.githubusercontent.com/30235603/202146670-d596d610-ae78-4b21-9969-4e73c132dc01.png)



In the labeling process, this is what the format must be. I need to prepare the data in this way and center position of X, center position of Y these refers to the bounding box and width and height of the bounding box.

#### 5.1- Data Preparation

First, I am starting with importing the libraries that I am going to use and read my data which is **label.csv** using **pd.read_csv()**

Then, I will get the information inside of the **xml** file which I have created before. 


![53](https://user-images.githubusercontent.com/30235603/202146673-902d96b8-7b3c-4ee1-a55d-29d5a92b0f56.png)


Using **xml** library I got the information which are **filename, width and height** then I combine them with the **df**.

In the next step, I will calculate **center_X, center_Y, width and height**.

![54](https://user-images.githubusercontent.com/30235603/202146675-4caf978d-61a2-44c1-a614-ac342c091e83.png)


Also, the folder structure of YOLO must be like this:

```bash
├── data
│   ├── train
│   │   ├── 001.jpeg
│   │   ├── 002.jpeg
│   │   ├── 001.txt
│   │   ├── 002.txt

│   ├── test
│   │   ├── 101.jpeg
│   │   ├── 102.jpeg
│   │   ├── 101.txt
│   │   ├── 102.txt
```

According to this schema, I need to create two folders which are called **train** and **test**. Inside of **train** I shoul put all images and their label information.

First, I will split the **dataframe** into **train** and **test**. I have 225 images, 200 images will be in the training folder, others will be in the test folder.


![55](https://user-images.githubusercontent.com/30235603/202146677-5adeab53-6536-493a-b81b-d4156af4c408.png)


Then I will copy every images inside of the folders.

![56](https://user-images.githubusercontent.com/30235603/202146682-b01490cd-f900-4040-98a8-d06c0948a460.png)

![57](https://user-images.githubusercontent.com/30235603/202146684-4caa1dab-36c9-450b-b066-66723d79b3ee.png)


For **test** folder, whole steps are same, only change train to **test**.


As next step, I will also create a **yaml** file. It is required for training process.


![58](https://user-images.githubusercontent.com/30235603/202146688-2e113701-da48-421a-baa3-577b665caedc.png)



---


#### 5.2- Training


### **NOTE:**
I have trained a new model using **YOLOv5** in **Google Colab** due to the **free GPU Service**. 
Now, I will explain what and why I did during the process.

As first step, I opened a new notebook and set my current working directory. After these I checked if I am in correct directory which I was.
Then I cloned YOLOv5 repo from GitHub and then I installed the requirements.

![59](https://user-images.githubusercontent.com/30235603/202146693-34c28404-b345-4dc1-8700-8b0af0d6e0f6.png)


After the installation, I changed my current working directory ad I set it inside of **yolov5** file.
Then, by typing this magical word, the training process started

```python
!python train.py --data data.yaml --cfg yolov5s.yaml --batch-size 8 --name Model --epochs 100
```


![60](https://user-images.githubusercontent.com/30235603/202146700-4788462c-2b5f-461c-82f4-f83ad0cee834.png)


![61](https://user-images.githubusercontent.com/30235603/202149664-d0f4950e-ddc3-4fa2-aed6-5d5709526e45.png)


As you can see in this detail, 100 epochs took 0.720 hours, also best model has saved into there: **runs/train/Model/weights/best.pt**

Then in order to use the model, I exported it as **onnx** format using it with OpenCV.

![62](https://user-images.githubusercontent.com/30235603/202149666-ce437017-daeb-4f3c-a12f-38479f75b9b1.png)


I would also like to share some images from validation.


![63](https://user-images.githubusercontent.com/30235603/202149677-35e34dff-b086-4a88-bb26-2a43b4573b23.png)

![64](https://user-images.githubusercontent.com/30235603/202149699-74eb6dc5-5959-44d8-9bbb-2504282ca129.png)

![65](https://user-images.githubusercontent.com/30235603/202149713-c15a30d7-ce81-4b14-90ab-85a0f5670a5e.png)


Results are much much much better! 

I also want to share **PR_Curve** which is **Precision Curve**

![66](https://user-images.githubusercontent.com/30235603/202149726-2edcf5af-e392-4b12-89bd-d8103646e11f.png)


#### 5.3- Using the Trained YOLO Model and Prediction

In this part, I will define some functions for prediction using the trained YOLO Model, which I did.

I am starting with defining the sizes. That’s what the YOLO Model uses.

![67](https://user-images.githubusercontent.com/30235603/202149729-15ae4640-96df-4cab-904a-c3e8b090a666.png)


Here, I will load **YOLO Model** using **OpenCV** functions.

![68](https://user-images.githubusercontent.com/30235603/202149730-8e3b8025-583e-439a-8ffc-038d7d43df0b.png)


Then, I will resize the image by adding **np.zeros** as what YOLO format requires.

![69](https://user-images.githubusercontent.com/30235603/202149736-72f1216a-33aa-47be-9b10-2e1cb90cbc5c.png)


And, before the final step, I will get the predictions

![70](https://user-images.githubusercontent.com/30235603/202149739-a8a7e9d4-4024-4be0-bf4c-d589d3a8379d.png)


But there is something that I must warn!

Totally, detection has 6 feature which are **center_x, center_y, w, h, confidence, probability**.

By using these features, I will filter **detection** based on **confidence probability score**

![71](https://user-images.githubusercontent.com/30235603/202149742-a4374c60-7100-4038-9029-eb7240b6b7c6.png)


Here is outputs,

![72](https://user-images.githubusercontent.com/30235603/202149743-bf4189c5-9264-44a2-af9a-2755de126e47.png)


But now they are in **np.array** format, I should turn them into **list**

![73](https://user-images.githubusercontent.com/30235603/202149744-f5177234-c1f6-47ce-8391-a29fcbcfa78d.png)


As the final step, I have to apply **Non Maximum Suppression** on **Bounding Box** using **OpenCV**

![74](https://user-images.githubusercontent.com/30235603/202149746-3726640f-3e90-4a8e-b7db-32e4fbcd3729.png)


And, I am ready for drawing **Bounding Box**

![75](https://user-images.githubusercontent.com/30235603/202149748-7fd914ec-2cdd-493e-872e-f6bffcc67e80.png)


He we go, it looks great! As we can realize easily, **YOLO** did great job!


#### 5.4- Editing the Functions (Clean Code)

Now, what I will do is that I will put all steps together.

![76](https://user-images.githubusercontent.com/30235603/202149755-1d7199bc-5f52-4955-b02a-6d46909c605b.png)

![77](https://user-images.githubusercontent.com/30235603/202149762-5e12188d-5361-40f8-983d-4fc91fd99ec4.png)

![78](https://user-images.githubusercontent.com/30235603/202149764-69c74521-0e8a-4ab4-b90e-3963dd16cc66.png)

![79](https://user-images.githubusercontent.com/30235603/202149768-32b1b1a0-2a25-410e-937d-3845ca5f756d.png)


Now all functions are created, and they are ready for the test!

![80](https://user-images.githubusercontent.com/30235603/202149771-3a9d7b11-8dac-41d8-9f89-e1955ff150f3.png)


In the another image, it also did great job!

#### 5.5- Extract Text with PyTesseract

Now I will apply **pytesseract** in order to extract text but I will do this separately from these function that I have created. I will define another functions for these steps.

For this, I am defining one more function called **extract_text**

![81](https://user-images.githubusercontent.com/30235603/202149776-69e4940a-8020-44c6-9320-5a130a5e11c1.png)


And I will use this function where drawings are happening, which is **drawings** function. Because in order to get text from **license plate** I have to have **roi** here, it means **bounding box**. 
After getting text, I will show them in the image, it will be inside of a rectangle.

![82](https://user-images.githubusercontent.com/30235603/202149777-4c2d53fd-04aa-4bea-8998-65ac51a7453e.png)


And I will update the **prediction function** due to that I changed **drawings function**

![83](https://user-images.githubusercontent.com/30235603/202149779-d7244902-5046-40f0-90fc-b358e101b900.png)

And **result**:


![84](https://user-images.githubusercontent.com/30235603/202149783-3b62d313-8097-4b79-bca8-6668b55c26e7.png)

![85](https://user-images.githubusercontent.com/30235603/202149654-6a0d8941-e88e-4ef0-9530-7194778e8ff5.png)
