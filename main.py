from pygta5.grabscreen import grab_screen
import cv2
from RoadDetector import MyKeras
from keras.utils import to_categorical
import numpy as np

if __name__ == '__main__':
    model, _ = MyKeras.load_latest_model(model_load_dir='./RoadDetector/models/main')

    cv2.namedWindow('screen')
    cv2.moveWindow('screen', 1280, 0)

    while True:
        screen = grab_screen([0, 26, 0+1280, 26+720])
        rgb = cv2.resize(screen, (512, 288))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        y = model.predict(np.array([rgb])/255.0)
        y = np.argmax(y, axis=-1)
        y = np.array(to_categorical(y, num_classes=3)[0]).astype(np.uint8)*255
        y[:,:,0] = 0
        y_bgr = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)

        added = cv2.add(bgr, y_bgr)

        cv2.imshow('screen', added)

        if cv2.waitKey(25) & 0xFF == ord('q'):
           cv2.destroyAllWindows()
           break
