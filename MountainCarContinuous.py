import random
import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import time

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
    def GetValue(self,iObservations):
        oSelectedAction=random.randint(-200,200)
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
    def GetValue(self,iObservations):
        #position and velocity for ease
        position, velocity = iObservations
        action = 0.0
        if position < -0.9:
            action = 1
        elif position > 0.4:
            action = 1
        elif position < -0.6 and velocity < -0.03:
            action = -0.5
        elif position < -0.6 and velocity > -0.01:
            action = 0.5
        elif position < -0.3:
            action = -0.5
        elif position < 0.0:
            action = 1
        elif position < 0.2 and velocity < -0.03:
            action = -0.5
        elif position < 0.2:
            action = 0.5
        elif position < 0.4:
            action = 1
        return action

#Bot That Uses A Simple Network
class BasicNNBotMountainCarContinuous():
    #Parameter Used In eleminating bad games
    cGOODGAMESCORETHRESHOLD=80
    #File Path for Saving the Trained Model For Avoiding Retraining It
    cLOADBOTPATH="BasicNNBotMountainCarContinuousModel.h5"
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
            self.mModel.add(Dense(128, activation='relu', input_dim=self.mObservationSpaceSize))
            self.mModel.add(Dense(64, activation='relu'))
            self.mModel.add(Dense(32, activation='relu'))
            self.mModel.add(Dense(16, activation='relu'))
            self.mModel.add(Dense(1, activation='linear')) 
            self.mModel.compile(loss='mse', optimizer=Adam(), run_eagerly=True)
            self.mDoesRequireTraining=True
    def Train(self):
        #Run The Game 5000 Times With RandomBot
        vMountCarEnv=MountainCarContinousWrapper()
        vRuleBasedBot=RuleBasedBot(self.mObservationSpaceSize, self.mActionSpaceSize)
        vRandomBot=RandomBot(self.mObservationSpaceSize, self.mActionSpaceSize)
        vAllActions=vMountCarEnv.Run(5000,False,False,vRuleBasedBot)
        #Get The Data From The Good Games
        vTrainingDataFeatures=[]
        vTrainingDataResults=[]
        for vActions in vAllActions:
            if(vActions["Score"]>self.cGOODGAMESCORETHRESHOLD):
                vTrainingDataFeatures=vTrainingDataFeatures+vActions["Observations"]
                vTrainingDataResults=vTrainingDataResults+vActions["Values"]
        #Input Had To Be A NumPy Array
        vTrainingData=np.array(vTrainingDataFeatures)
        vTrainingLabels=np.array(vTrainingDataResults)
        #This Is Where Model IS Trained
        self.mModel.fit(vTrainingData,vTrainingLabels,epochs=3, batch_size=32)
        #Saving The Trained Model
        self.mModel.save(self.cLOADBOTPATH)
        self.mDoesRequireTraining=False
    def GetValue(self,iObservations):
        #Ask Network For An Output
        vPredictionResult=self.mModel.predict(np.array([iObservations]))[0][0]
        #If The Output is Closer to the 0 then Cart Moves Left Else Cart Moves Right
        if(vPredictionResult>=0.8):
            oSelectedAction = 1
        elif (vPredictionResult >= 0.6):
            oSelectedAction = 0.5
        elif (vPredictionResult >= 0.4):
            oSelectedAction = 0.0
        elif (vPredictionResult >= 0.2):
            oSelectedAction = -0.5
        else:
            oSelectedAction = -1
        return oSelectedAction
    

class MountainCarContinousWrapper():
    cENVNAME="MountainCarContinuous-v0"
    def __init__(self):
        self.mProblemEnvironment=gym.make(self.cENVNAME)
        self.mObservationSpaceSize=self.mProblemEnvironment.observation_space.shape[0]
        self.mActionSpaceSize=self.mProblemEnvironment.action_space.shape[0]

    def Run(self,iIterationCount,iIsDisplayingGame,iIsDisplayingText,iBot):
        oAllActions=[]
        for vIterationIndex in range(iIterationCount):
            #Every Iteration We Start From The Beginning
            vCurrentObservation=self.mProblemEnvironment.reset()
            vIterationScore=0
            vIterationActions={"Observations":[],"Values":[],"Score":0}
            while True:
                #This Is The display
                if(iIsDisplayingGame):
                    self.mProblemEnvironment.render()
                #Get Bots Action
                vSelectedValue=iBot.GetValue(vCurrentObservation)
                #Apply action to Get To Next State
                vNewObservation,vReward,vIsDone,_=self.mProblemEnvironment.step([vSelectedValue])
                vIterationScore=vIterationScore+vReward
                #If Failed 
                if(vIsDone):
                    break
                #Save State And Action Taken 
                vIterationActions["Observations"].append(vCurrentObservation)
                vIterationActions["Values"].append([vSelectedValue])
                vCurrentObservation=vNewObservation
            #Save Score
            vIterationActions["Score"]=vIterationScore
            oAllActions.append(vIterationActions)
            #This Will Display Text For Each Trial
            if(iIsDisplayingText):
                print("Iteration Number: "+str(vIterationIndex)+" Achieved Score: "+str(vIterationScore))
        #Return OBservation, Action and Scores for Further Processing If Required
        return oAllActions

def main(iBotType):
    if(iBotType==None):
        return
    #Set Up the Environment
    vEnv=MountainCarContinousWrapper()
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
    vBotType=BasicNNBotMountainCarContinuous
    main(vBotType)
