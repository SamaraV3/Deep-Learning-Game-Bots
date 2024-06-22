import random
import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import tensorflow as tf
import time

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
#Bot That Return a Random Action
class RandomBot():
    def __init__(self,iObservationSpaceSize,iActionSpaceSize):
        #Action Space Size is Required So Bot Can Randomize Action
        self.mActionSpaceSize=iActionSpaceSize
        #Random Bot Does not Require Training
        self.mDoesRequireTraining=False
    #Random Bot Does not Require Training But Train Function Need To Be Present
    def Train(self):
        pass
    #This Is Where Bot Selects Random Action
    def GetAction(self,iObservations):
        oSelectedAction=random.randint(0,self.mActionSpaceSize-1)
        return oSelectedAction

#Bot that uses rules
class RuleBasedBot():
    def __init__(self,iObservationSpaceSize,iActionSpaceSize):
        #Action Space Size is Required So Bot Can Randomize Action
        self.mActionSpaceSize=iActionSpaceSize
        #Random Bot Does not Require Training
        self.mDoesRequireTraining=False
    #Random Bot Does not Require Training But Train Function Need To Be Present
    def Train(self):
        pass
    #This Is Where Bot Selects Random Action
    def GetAction(self,iObservations):

        #position and velocity for ease
        position, velocity = iObservations
        action = 1  # Default: do nothing
        #position thresholds from negative to positive:
        # -1.2    -0.9   -0.6  -0.3    0.0     0.2     0.4    0.6
        #smallest posible is -1.2 and largest possible is 0.6
        #velocity thresholds from negative to positive
        # -0.07   -0.035   0.0   0.035   0.07
        #-0.07 is the lowest velocity, 0.07 is the highest
        if position < -0.9:
            action = 2
        elif position > 0.4:
            action = 2
        elif position < -0.6 and velocity < -0.03:
            action = 0
        elif position < -0.6 and velocity > -0.01:
            action = 2
        elif position < -0.3:
            action = 0
        elif position < 0.0:
            action = 2
        elif position < 0.2 and velocity < -0.03:
            action = 0
        elif position < 0.2:
            action = 2
        elif position < 0.4:
            action = 2
        return action
            


#Bot That Uses A Simple Network
class BasicNNBotMountainCar():
    #Parameter Used In eleminating bad games
    cGOODGAMESCORETHRESHOLD=-199
    #File Path for Saving the Trained Model For Avoiding Retraining It
    cLOADBOTPATH="BasicNNBotMountainCarModel.h5"
    def __init__(self,iObservationSpaceSize,iActionSpaceSize):
        #Observations are The Input For Our Network So Their Size Is Improtant 
        self.mObservationSpaceSize=iObservationSpaceSize
        self.mActionSpaceSize=iActionSpaceSize
        #Checks if we already have a Pre-Trained Modal Present
        if(os.path.isfile(self.cLOADBOTPATH)):
            #IF There is Use that Modal
            self.mModel = keras.models.load_model(self.cLOADBOTPATH)
            self.mDoesRequireTraining=False
        else:
            #If Not Generate Model And Set The Training Flag
            #Simple Model With Fully Connected Layers: Input Is The Observations Output is a Value between 0 To 1
            self.mModel = Sequential()
            self.mModel.add(Dense(32, activation='relu', input_dim=self.mObservationSpaceSize))
            self.mModel.add(Dense(16, activation='relu'))
            self.mModel.add(Dense(1, activation='linear')) 
            self.mModel.compile(loss='mse', optimizer=Adam())
            self.mDoesRequireTraining=True
    def Train(self):
        #Run The Game 5000 Times With RandomBot
        vMountCarEnv=MountainCarWrapper()
        vRandomBot=RandomBot(self.mObservationSpaceSize,self.mActionSpaceSize)
        vRuleBasedBot=RuleBasedBot(self.mObservationSpaceSize, self.mActionSpaceSize)
        vAllActions=vMountCarEnv.Run(5000,False,False,vRuleBasedBot)
        #Get The Data From The Good Games
        vTrainingDataFeatures=[]
        vTrainingDataResults=[]
        for vActions in vAllActions:
            if(vActions["Score"]>self.cGOODGAMESCORETHRESHOLD):
                vTrainingDataFeatures=vTrainingDataFeatures+vActions["Observations"]
                vTrainingDataResults=vTrainingDataResults+vActions["Actions"]
        #Input Had To Be A NumPy Array
        vTrainingData=np.array(vTrainingDataFeatures)
        vTrainingLabels=np.array(vTrainingDataResults)
        #This Is Where Model IS Trained
        self.mModel.fit(vTrainingData,vTrainingLabels,epochs=10, batch_size=32)
        #Saving The Trained Model
        self.mModel.save(self.cLOADBOTPATH)
        self.mDoesRequireTraining=False
    def GetAction(self,iObservations):
        #Ask Network For An Output
        vPredictionResult=self.mModel.predict(np.array([iObservations]))[0][0]
        #If The Output is Closer to the 0 then Cart Moves Left Else Cart Moves Right
        if(vPredictionResult>=0.7):
            oSelectedAction=2
        elif (vPredictionResult>=0.4):
            oSelectedAction = 1
        else:
            oSelectedAction=0
        return oSelectedAction

class MountainCarWrapper():
    cENVNAME="MountainCar-v0"
    def __init__(self):
        self.mProblemEnvironment=gym.make(self.cENVNAME)
        self.mObservationSpaceSize=self.mProblemEnvironment.observation_space.shape[0]
        self.mActionSpaceSize=self.mProblemEnvironment.action_space.n

    def Run(self,iIterationCount,iIsDisplayingGame,iIsDisplayingText,iBot):
        oAllActions=[]
        for vIterationIndex in range(iIterationCount):
            #Every Iteration We Start From The Beginning
            vCurrentObservation=self.mProblemEnvironment.reset()
            vIterationScore=0
            vIterationActions={"Observations":[],"Actions":[],"Score":0}
            while True:
                #This Is The display
                if(iIsDisplayingGame):
                    self.mProblemEnvironment.render()
                #Get Bots Action
                vSelectedAction=iBot.GetAction(vCurrentObservation)
                #Apply action to Get To Next State
                vNewObservation,vReward,vIsDone,_=self.mProblemEnvironment.step(vSelectedAction)
                #If Failed 
                if(vIsDone):
                    break
                vIterationScore=vIterationScore+vReward
                #Save State And Action Taken 
                vIterationActions["Observations"].append(vCurrentObservation)
                vIterationActions["Actions"].append([vSelectedAction])
                vCurrentObservation=vNewObservation
            #Save Score
            vIterationActions["Score"]=vIterationScore
            oAllActions.append(vIterationActions)
            #This Will Display Text For Each Trial
            if(iIsDisplayingText):
                print("Iteration Number: "+str(vIterationIndex)+" Achieved Score: "+str(vIterationScore))
        #Return Observation, Action and Scores for Further Processing If Required
        return oAllActions

def main(iBotType):
    #Set Up the Environment
    vEnv=MountainCarWrapper()
    #Set Up the Bot
    vBot=iBotType(vEnv.mObservationSpaceSize,vEnv.mActionSpaceSize)
    #If the Bot Requires Training Call Train Function To Train  
    if(vBot.mDoesRequireTraining):
        print("Training The Bot")
        vBot.Train()
        print("Training Complete")
    #Test The Bot
    print("Testing The Bot")
    #Run Code 200 Times With Text And Image Display Using The Provided Bot
    vAllActions=vEnv.Run(200,True,True,vBot)
    print("Testing Complete")
    #Calculate Average Score
    vTotalScore=0
    for vActions in vAllActions:
        vTotalScore=vTotalScore+vActions["Score"]
    print("Average Score: "+str(vTotalScore/len(vAllActions)))

if __name__ == "__main__":
    main(BasicNNBotMountainCar)
