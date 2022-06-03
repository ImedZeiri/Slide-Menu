from kivy.app import App
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
import os
import cv2
import gtts
import mediapipe as mp
import numpy as np

Window.size = (350,600)



kv = '''
ScreenManager:
    ListScreen:
    Trainning:


<ListScreen>
    name:'listscreen'
    PageLayout:
        BoxLayout:
            canvas:
                Color:
                    rgba: 216/255., 195/255., 88/255., 1
                Rectangle:
                    pos: self.pos
                    size: self.size
           
    
            orientation: 'vertical'
        BoxLayout:
            orientation: 'vertical'
            canvas:
                Color:
                    rgba: 109/255., 8/255., 57/255., 1
                Rectangle:
                    pos: self.pos
                    size: self.size
            Button:
                text: 'Dorseu'
                halign:'center'
                height: '38dp'
                pos_hint: {'center_y':0.7}
                background_color: 0, 0, 0, 0
                on_press:
                    root.manager.current = 'trainning'
                    root.manager.transition.direction = 'down'
                
                
        BoxLayout:
            orientation: 'vertical'
            canvas:
                Color:
                    rgba: 39/205., 8/255., 57/255., 1
                Rectangle:
                    pos: self.pos
                    size: self.size
            Button:
                text: 'Poitrine'
                halign:'center'
                height: '38dp'
                pos_hint: {'center_y':0.7}
                background_color: 0, 0, 0, 0
                on_press:
                    root.manager.current = 'trainning'
                    root.manager.transition.direction = 'down'
                
        BoxLayout:
            orientation: 'vertical'
            canvas:
                Color:
                    rgba: 39/105., 8/155., 57/155., 1
                Rectangle:
                    pos: self.pos
                    size: self.size
            Button:
                text: 'Jambes'
                halign:'center'
                height: '38dp'
                pos_hint: {'center_y':0.7}
                background_color: 0, 0, 0, 0
                on_press:
                    root.manager.current = 'trainning'
                    root.manager.transition.direction = 'down'
                
        BoxLayout:
            orientation: 'vertical'
            canvas:
                Color:
                    rgba: 36/205., 37/205., 39/205., 1
                Rectangle:
                    pos: self.pos
                    size: self.size
            Button:
                text: 'Les Epaules'
                halign:'center'
                height: '38dp'
                pos_hint: {'center_y':0.7}
                background_color: 0, 0, 0, 0
                on_press:
                    root.manager.current = 'trainning'
                    root.manager.transition.direction = 'down'
            
                    
        BoxLayout:
            orientation: 'vertical'
            canvas:
                Color:
                    rgba: 139/105., 108/155., 107/155., 1
                Rectangle:
                    pos: self.pos
                    size: self.size
            Button:
                text: 'Autres muscles'
                halign:'center'
                height: '38dp'
                pos_hint: {'center_y':0.7}
                background_color: 0, 0, 0, 0
                on_press:
                    root.manager.current = 'trainning'
                    root.manager.transition.direction = 'down'
                    

<Trainning>
    name:'trainning'
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (940, 980)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not(camera.play)
        size_hint_y: None
        height: '50dp'
    Button:
        text: 'Back'
        size_hint_y: None
        height: '25dp'
        on_press:
            root.manager.current = 'listscreen'
            root.manager.transition.direction = 'up'                    

     
'''


class ListScreen(Screen):
    pass


class Trainning(Screen):
    pass

class TestApp(App):
    def build(self):
        # Create the screen manager
        sm = ScreenManager()
        sm.add_widget(ListScreen(name='listscreen'))
        sm.add_widget(Trainning(name='trainning'))
        return sm

    def train(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        def calculate_angle(a, b, c):
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle

        cap = cv2.VideoCapture(0)

        # Curl counter variables
        counter = 0
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    print([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]])


                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage == 'down':
                        stage = "up"
                        counter += 1
                        if counter == 10:
                            t1 = gtts.gTTS("good job , we will move to the next exercice")
                            t1.save("job.mp3")
                            file = "job.mp3"
                            os.system("mpg123 " + file)
                        print(counter)
                except:
                    pass

                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

                cv2.putText(image, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Stage data
                cv2.putText(image, 'STAGE', (65, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    runTouchApp(Builder.load_string(kv))


