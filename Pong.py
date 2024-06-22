import random
import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import time
import cv2
import tensorflow as tf

#Bot That Return a Random Action For Pong Game
class PongRandomBot():
    def __init__(self,iObservationSpaceSize,iActionSpaceSize):
        #Action Space Size is Required So Bot Can Randomize Action
        self.mActionSpaceSize=iActionSpaceSize
        #Random Bot Does not Require Training
        self.mDoesRequireTraining=False
    #Random Bot Does not Require Training But Train Function Need To Be Present
    def Train(self):
        pass
    #This Is Where Bot Selects Random Action
    #There Are 2 Actions For The Pong 3 and 4
    def GetAction(self,iObservations):
        oSelectedAction=random.randint(0,self.mActionSpaceSize-1)+2
        return oSelectedAction

class PongRuleBasedBot():
    def __init__(self, iObservationSpaceSize, iActionSpaceSize):
        self.mActionSpaceSize = iActionSpaceSize
        self.mDoesRequireTraining=False
        self.prev_observation = None
    def Train(self):
        pass
    def GetAction(self, iObservations):
        #first I will convert the image to grayscale
        my_img = iObservations
        my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
        my_img = my_img[35:]#this gets rid of score area and white at bottom of screen
        #Acc I cannot cut it bcuz then there's a problem finding the green I need
        #so i just need to ignore the areas that would be cut when I am getting the pong location
        #a solution coult be to cut my image AFTER I've detected the paddle
        copy = my_img.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)#used to make my paddle white
        lower=np.array([50, 100,100])
        upper=np.array([70, 255, 255])
        # Defining mask for detecting color
        mask = cv2.inRange(copy, lower, upper)
        #FOR THE MASK IMAGE ONLY THE GREEN IS COLORED IN WHITE
        #USE THIS TO LATER COMPARE TO THE WHITE BALL
        # Display Image and Mask
        #iterate through mask to find highest white corner of mask

        #my_img = my_img[35:185]#this gets rid of score area and white at bottom of screen
        #cutting it l
        paddle_coords = []
        for h in range(mask.shape[0]):
            foundCorner = False
            for w in range(my_img.shape[1]):
                #get the pixel values at that location on mask
                corner_paddle = mask[h,w]
                #check if it is white
                if np.array_equal(corner_paddle, 255):
                    foundCorner = True
                    paddle_coords.append([h,w])
                    break
            if foundCorner:
                break
        #try the following -- it works!
        if len(paddle_coords) == 1:
            if paddle_coords[0][0]+ 15 >= my_img.shape[0]:
                paddle_coords.insert(0, [paddle_coords[0][0]-15, paddle_coords[0][1]])#top corner
            else:
                paddle_coords.append([paddle_coords[0][0]+15, paddle_coords[0][1]])#bottom corner
        #paddle_coords[0] returns the higest corner while paddle_coords[1] returns the lowest one
        ball_coords = []
        #iterate through my image to find white. I just wanna see if any pixel is white before i continue, in case the ball rlly isnt white
        for h in range(150):
            foundCorner = False
            for w in range(my_img.shape[1]):
                pixel = my_img[h,w]
                if np.all(np.greater_equal(pixel, [230, 230, 230])):
                    ball_coords.append([h,w])
                    foundCorner = True
                    break
            if foundCorner:
                break
        #check if the pixel above is the same color as the current pixel
        #if it is, add 3. Otherwise subtract 3 from the y
        #This Works!
        if len(ball_coords) == 1:
            if np.array_equal(my_img[ball_coords[0][0], ball_coords[0][1]], my_img[ball_coords[0][0]-1, ball_coords[0][1]]):
                #means we got a bottom corner.
                ball_coords.insert(0, [ball_coords[0][0]-3, ball_coords[0][1]])#top corner
            else:
                ball_coords.append([ball_coords[0][0]+3, ball_coords[0][1]])#bottom corner
        #Now comes comparisons that I will use to dictate whether to move the paddle up or down
        #if ball[0] y value (ball[0][0]) is greater than paddle[1][0], then move down (down = 3)
        #paddle[0] = top corner paddle[1] = bottom corner
        #ball[0] = top corner ball[1] = bottom corner
        if len(ball_coords) == 2 and len(paddle_coords) == 2:
            if ball_coords[1][0] <= paddle_coords[1][0]-3 and ball_coords[0][0] >= paddle_coords[0][0]+3:
                if self.prev_observation is not None:
                    if self.prev_observation == 3:
                        self.prev_observation = 2
                        return 2
                    else:
                        self.prev_observation = 3
                        return 3
                else:
                    self.prev_observation = random.randint(2,3)
                    return self.prev_observation
            elif ball_coords[1][0] > paddle_coords[1][0]:
                self.prev_observation = 3
                return 3
            elif ball_coords[0][0] < paddle_coords[0][0]:
                self.prev_observation = 2
                return 2
            else:
                if self.prev_observation is not None:
                    if self.prev_observation == 3:
                        self.prev_observation = 2
                        return 2
                    else:
                        self.prev_observation = 3
                        return 3
                else:
                    self.prev_observation = random.randint(2,3)
                    return self.prev_observation
        else:
            if self.prev_observation is not None:
                    if self.prev_observation == 3:
                        self.prev_observation = 2
                        return 2
                    else:
                        self.prev_observation = 3
                        return 3
            else:
                self.prev_observation = random.randint(2,3)
                return self.prev_observation


#Bot That Uses A Simple Network
class BasicNNBotPong():
    #Parameter Used In eleminating bad games
    cGOODGAMESCORETHRESHOLD=0#Only uses the games I won
    #File Path for Saving the Trained Model For Avoiding Retraining It
    cLOADBOTPATH="BasicNNBotPongModel.h5"
    def __init__(self,iObservationSpaceSize,iActionSpaceSize):
        #Observations are The Input For Our Network So Their Size Is Improtant 
        self.mObservationSpaceSize=3
        self.mActionSpaceSize=iActionSpaceSize
        self.prev_observation = None
        #Checks if we already have a Pre-Trained Modal Present
        if(os.path.isfile(self.cLOADBOTPATH)):
            #IF There is Use that Modal
            self.mModel = keras.models.load_model(self.cLOADBOTPATH)
            self.mDoesRequireTraining=False
        else:
            #If Not Generate Model And Set The Training Flag
            #Simple Model With Fully Connected Layers: Input Is The Observations Output is a Value between 0 To 1
            self.mModel = Sequential()
            #self.mModel.add(Dense(32, activation='relu', input_dim=self.mObservationSpaceSize))
            self.mModel.add(Dense(1024, activation='relu', input_dim=24))   
            self.mModel.add(Dense(512, activation='relu'))
            self.mModel.add(Dense(128, activation='relu'))
            self.mModel.add(Dense(64, activation='relu'))
            self.mModel.add(Dense(32, activation='relu'))
            self.mModel.add(Dense(16, activation='relu'))
            self.mModel.add(Dense(8, activation='relu'))
            self.mModel.add(Dense(4, activation='relu'))
            self.mModel.add(Dense(1, activation='linear')) 
            self.mModel.compile(loss='mse', optimizer=Adam())
            #self.mModel.summary()
            #config = self.mModel.get_config()
            #print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels)
            self.mDoesRequireTraining=True
    def Train(self):
        #Run The Game 5000 Times With RandomBot
        vPongEnv=PongEnvWrapper()
        vRandomBot=PongRandomBot(self.mObservationSpaceSize,self.mActionSpaceSize)
        vRuleBasedBot=PongRuleBasedBot(self.mObservationSpaceSize, self.mActionSpaceSize)
        vAllActions=vPongEnv.Run(True,True,vRuleBasedBot)
        #Get The Data From The Good Games
        vTrainingDataFeatures=[]
        vTrainingDataResults=[]
        for vActions in vAllActions:
            if(vActions["Score"]>self.cGOODGAMESCORETHRESHOLD):
                vTrainingDataFeatures=vTrainingDataFeatures+vActions["Observations"]
                vTrainingDataResults=vTrainingDataResults+vActions["Actions"]
        #Input Had To Be A NumPy Array
        vTrainingData = np.array(vTrainingDataFeatures)
        vTrainingLabels=np.array(vTrainingDataResults)
        #train_dataset = tf.data.Dataset.from_tensor_slices((vTrainingData, vTrainingLabels))
        #This Is Where Model IS Trained
        self.mModel.fit(vTrainingData, vTrainingLabels,epochs=10, batch_size=100)
        #Saving The Trained Model
        self.mModel.save(self.cLOADBOTPATH)
        self.mDoesRequireTraining=False
    def GetAction(self,iObservations):
        #Ask Network For An Output
        temp = iObservations.copy()
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        temp = temp[35:190]#crop to get rid of the white above and below
        temp[temp >= 100] = 255#makes the ball and paddles white
        temp[temp != 255] = 0#makes everything else black
        contour_temp = temp.copy()
        contours, hierarchy = cv2.findContours(contour_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        obs_list = []
        for contour in contours:
            temp = []
            for point in contour:
                for value in point:
                    x, y = value.tolist()
                    obs_list.append(x)
                    obs_list.append(y)
        if len(obs_list) == 24:
            obs_list = np.array(obs_list)
            obs_list = np.reshape(obs_list, (-1, 24))
            vPredictionResult=self.mModel.predict(np.array(obs_list))[0][0]
            #If The Output is Closer to the 0 then Cart Moves Left Else Cart Moves Right
            if(vPredictionResult>2.5): 
                oSelectedAction=3
                self.prev_observation=3
            else:
                oSelectedAction = 2
                self.prev_observation = 2
            return oSelectedAction
        else:
            if self.prev_observation is None:
                self.prev_observation = random.randint(2,3)
            elif self.prev_observation == 3:
                self.prev_observation = 2
            else:
                self.prev_observation = 3
            return self.prev_observation

def main(iBotType):
    #Set Up the Environment
    vGameEnv=PongEnvWrapper()
    #Set Up the Bot
    vBot=iBotType(vGameEnv.mObservationSpaceSize,2)
    #If the Bot Requires Training Call Train Function To Train  
    if(vBot.mDoesRequireTraining):
        print("Training The Bot")
        vBot.Train()
        print("Training Complete")
    #Test The Bot
    print("Testing The Bot")
    #Run Game With Text And Image Display Using The Provided Bot
    vAllActions=vGameEnv.Run(True,True,vBot)
    print("Testing Complete")

class PongEnvWrapper():
    cENVNAME="Pong-v0"
    def __init__(self):
        self.mProblemEnvironment=gym.make(self.cENVNAME)
        self.mObservationSpaceSize=self.mProblemEnvironment.observation_space.shape[0]
        self.mActionSpaceSize=self.mProblemEnvironment.action_space.n

    def Run(self,iIsDisplayingGame,iIsDisplayingText,iBot):
        oAllActions=[]
        #We Start From The Beginning
        vCurrentObservation=self.mProblemEnvironment.reset()
        vScore=0
        vSetActions={"Observations":[],"Actions":[],"Score":0}
        n = 0
        #change the while True when submitting
        while True:
            #This Is The display
            if(iIsDisplayingGame):
                self.mProblemEnvironment.render()
            #Get Bots Action
            vSelectedAction=iBot.GetAction(vCurrentObservation)
            #Apply action to Get To Next State
            vNewObservation,vReward,vIsDone,_=self.mProblemEnvironment.step(vSelectedAction)
            #Save State And Action Taken 
            #COnvert image to grayscale
            temp = vCurrentObservation.copy()
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp[35:190]#crop to get rid of the white above and below
            temp[temp >= 100] = 255#makes the ball and paddles white
            temp[temp != 255] = 0#makes everything else black
            contour_temp = temp.copy()
            contours, hierarchy = cv2.findContours(contour_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #draw contours below
            #cv2.drawContours(contour_temp, contours, 0, (255, 255, 255), 1)
            #cv2.imshow("Temp", contour_temp)
            #contours returns a numpy array of (x,y) coordinates of boundary points of the object. Try using that as an observation
            obs_list = []
            for contour in contours:
                temp = []
                for point in contour:
                    for value in point:
                        x, y = value.tolist()
                        obs_list.append(x)
                        obs_list.append(y)
                #obs_list.append(temp)
            if len(obs_list) == 24:
                vSetActions["Observations"].append(obs_list)#This definitely appends the contours as a regular list
                vSetActions["Actions"].append([vSelectedAction])
            vCurrentObservation=vNewObservation
            #End Of Set There Are 20 Sets in The Game
            if(vReward!=0):
                if(iIsDisplayingText):
                    if(vReward==-1):
                        print("Set Ended, Set Lost")
                    else:
                        print("Set Ended, Set Won")
                    #Save State And Action Taken
                    n += 1
                    vScore=vScore+vReward
                    vSetActions["Score"]=vReward
                    oAllActions.append(vSetActions.copy())
            #If Game Ended
            if(vIsDone):
                break

        #This Will Display Text At The End Of Game
        if(iIsDisplayingText):
           print("Game Ended, Achieved Score: "+str(vScore))
        #Return Observation, Action and Scores for Further Processing If Required
        return oAllActions



if __name__ == "__main__":
    main(BasicNNBotPong)
