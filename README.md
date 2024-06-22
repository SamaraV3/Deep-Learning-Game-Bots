# Deep-Learning-Game-Bots
This repository contains files used to solve the Cart Pole, Mountain Car, and Continuous Mountain Car problems using both rule based and deep learning bots in tandem. it also contains a deep learning bot that is able to win the Pong Game at least once.

This project was an assignment for the First Year Research Immersion program course at Binghamton University. The goal was to develop deep learning bots capable of solving various problems using the Python Gym API. The problems tackled are:
- Cart Pole
- Mountain Car
- Mountain Car Continuous
- Pong

In this assignment, I was given template code for the Random Bot function present in all program files. I then had to create the RuleBasedBot function for each individual problem which could win the games a certain number of times. This rule based bot was then called by the deep learning bot to gather data to learn from, until the deep learning boss was able to play the games alone. I was given template code for the deep learning based bots for the first three problems, and had to edit parameters and requirements so they would be able to solve their respective problems. For the Pong Game I created both the rule based and deep learning based bots alone.

## Requirements:
- Python 3.6+
- OpenAI Gym
- Keras
- TensorFlow
- NumPy
- OpenCV
- Random

## File Descriptions
### CartPole.py
  - **RuleBasedBot:** Pushes the cart right if the pole is leaning left, and left if the pole is leaning right. If the pole is nearly vertical and is moving too fast, the bot slows down (goes in the opposite direction than it was going in before) to avoid overshooting.
  - **BasicNNBotCartPole:** Deep learning based bot. Runs the rule based bot 5000 times, and only uses games that achieve a score of over 60 for training. 
### MountainCar.py
  - **RuleBasedBot:** Pushes the car right if it is at a negative peak, positive peak, or if the position is greater than 0.2 (used as a threshold value). It also pushes right if the bot is somewhat high and negative but not at a peak, and the velocity seems more positive or weakly negative. In all other cases the car pushes left.
  - **BasicNNBotMountainCar:** Deep learning based bot. Runs the rule based bot 5000 times, and only uses games that achieve a score of over -199 (games that were won) for training.
### MountainCarContinuous.py
  - **RuleBasedBot:** The action function for this class uses similar rules to the one for MountainCar, but it returns values between 1 and -1 instead. It returns 1 and -1 when the car should use maximum power (when positions are incredibly high or in the trough of the mountain), and returns -0.5 or 0.5 if half power is called for instead.
  - **BasicNNBotMountainCarContinuous:** Deep learning based bot. Runs the rule based bot 5000 times, and only uses games that achieve a score of over 80.
### Pong.py
  - **PongRuleBasedBot:** First converts the observation image to grayscale. Then it detects the paddles and ball using HSV masking and contour detection. Then it moves the paddle (of the computer "player") based on the relative positions of the player's paddle and the ball.
  - **BasicNNBotPong:** Runs the rule based bot until one game is completed. The contours of the paddles and the ball are used as inputs for training. The first 10,000 results are then run through this bot for training, using the games the player won only.

## Issues
- The Pong Game takes a long time to run because of the usage of image conversion and contour detection. In addition, if the rule based bot is unable to win any game, then the deep learning bot does not work.
- The BasicNNBotPong doesn’t always play the game effectively. The paddle tends to fidget up and down when the ball is not higher or lower than both of its corners, and this sometimes causes the paddle to miss the ball. This is because there is no option for the paddle to not move, and to make up for this the paddle is programmed to move back and forth.
- There are some moments when only two contours are detectable on the screen. This occurs before the ball is loaded or when the ball touches a paddle, At these moments, actions cannot be chosen using the neural network as it required 12 points.
- These bots do not use the most efficient manners to win all these games. They were created with the knowledge I had accrued from the one semester, first year course that I created them during. There are better solutions for these games, however these solutions also work often.




