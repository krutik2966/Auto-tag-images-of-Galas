import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten,Dense,Dropout
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from PIL import Image
from PIL import ImageFilter

#data preprocessing
dataset = pd.read_csv('./dataset/train.csv')

data_array = dataset.iloc[:,:].values
cnt =[0]*4

for i in range(len(data_array)):
    if(data_array[i][1]=='Attire'):
        cnt[0] = cnt[0]+1
    elif(data_array[i][1]=='Decorationandsignage'):
        cnt[1]=cnt[1]+1
    elif(data_array[i][1]=='Food'):
        cnt[2]=cnt[2]+1
    else:
        cnt[3]=cnt[3]+1
        
total=len(data_array)
for i in range(total):
    if(((data_array[i][1]=='Attire') and (cnt[0]<2000)) or 
        ((data_array[i][1]=='Decorationandsignage') and (cnt[1]<2000)) or 
        ((data_array[i][1]=='misc') and (cnt[3]<2000))):
        
        img = Image.open('./dataset/Train Images/'+data_array[i][0]+'')
        img=img.convert("RGB")
        im_unsharp=img.filter(ImageFilter.UnsharpMask)
        im_unsharp.save('./dataset/Train Images/unsharp_'+data_array[i][0]+'')
        data_array=np.insert(data_array,2*len(data_array),['unsharp_'+data_array[i][0]+'',data_array[i][1]])
        data_array = data_array.reshape(-1,2)

        if(data_array[i][1]=='Attire'):
            cnt[0] = cnt[0]+1
        elif(data_array[i][1]=='Decorationandsignage'):        
            im_edge=img.filter(ImageFilter.EDGE_ENHANCE)
            im_edge.save('./dataset/Train Images/edge_'+data_array[i][0]+'')
            data_array=np.insert(data_array,2*len(data_array),['edge_'+data_array[i][0]+'','Decorationandsignage'])
            data_array = data_array.reshape(-1,2)
            cnt[1]=cnt[1]+2
        elif(data_array[i][1]=='misc'):
            cnt[3]=cnt[3]+1

img_ = data_array[:,0]
Y = data_array[:,1]

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
Y=Y.reshape(-1,1)
onehotencoder = OneHotEncoder(categories='auto')
Y = onehotencoder.fit_transform(Y).toarray()


train_image= []
for i in range(len(img_)):
    img = image.load_img('./dataset/Train Images/'+img_[i]+'',target_size=(128,128,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
    

X = np.array(train_image)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.07)

train_datagen = image.ImageDataGenerator(
   featurewise_center=True,
   featurewise_std_normalization=True,
   rotation_range=20,
   width_shift_range=0.3,
   height_shift_range=0.3, 
   zoom_range=0.3,
   shear_range=0.2,
   horizontal_flip=True
   )

x_tr,x_ts,y_tr,y_ts = train_test_split(X_train,y_train, test_size=0.2)

base_model = MobileNet(input_shape=(128,128,3), include_top=False,weights='imagenet')
base_model.trainable = False

model = Sequential([
                    base_model,
                    Flatten(),
                    Dense(units=512,activation='relu'),
                    Dropout(0.5),
                    Dense(units=4,activation='softmax')
])
model.summary()

checkpoint = ModelCheckpoint('model_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=13,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_detail = model.fit(train_datagen.flow(x_tr,y_tr,batch_size=32), 
                         steps_per_epoch=len(x_tr)/16,
                         epochs=300,
                         callbacks = callbacks,
                         validation_data=(x_ts,y_ts))

y_pred = model.predict(X_test)
y_pred = [y_pred[i].argmax() for i in range(len(y_pred))]
y_test = [y_test[i].argmax() for i in range(len(y_test))]
f1_score(y_test,y_pred,average='weighted')

test_data = pd.read_csv('./dataset/test.csv')
img_test = test_data.iloc[:,0].values

test_image= []
for i in range(len(img_test)):
    img = image.load_img('./dataset/Test Images/'+img_test[i]+'',target_size=(128,128,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)

test = np.array(test_image)
   
y_pred = model.predict(test)

ans = [y_pred[i].argmax() for i in range(len(y_pred))]
ans_str=[]
for i in range(len(ans)):
    if(ans[i]==0):
        ans_str.append('Attire')
    elif(ans[i]==2):
        ans_str.append('Food')
    elif(ans[i]==1):
        ans_str.append('Decorationandsignage')
    else:
        ans_str.append('misc')
        
np.savetxt("ans_str.csv",ans_str,fmt='%s',delimiter=",")





