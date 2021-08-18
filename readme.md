# gym-auv

A Python simulation framework for Collision Avoidance for Unmanned Surface Vehicle using Deep Reinforcement Learning.

The detailed explanation of the software structure can be found in Eivind Meyers repository [gym-auv](https://github.com/EivMeyer)

## Prerequisites
Note: Requires Python 3.7

Note: Pybullet needs Microsoft Visual C++ 14.0. Install it with "Build Tools for Visual Studio".

Note: Stable-Baselines only supports Tensorflow 1.14, Tensorflow 2 support is planned. 

! Install Microsoft MPI (https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (msmpisetup.exe , not SDK)

Note: Run the following __first__.
```
conda install -c conda-forge shapely
conda install swig
conda install ffmpeg
```

Then run 

```
pip install -e ./gym-auv/
```
## Running the code
You can now execute the script by running 
```
python run.py <mode> <env> <-modifier kwarg>
``` 
The run script can be executed with the -h flag for a comprehensive overview of the available usage modes.

### Examples:
Manual control (arrow keys), quit by pressing "q".
```
python run.py play TestScenario1-v0
```
Train a PPO agent in the MovingObstaclesNoRules environment.
```
python run.py train MovingObstaclesNoRules-v0
``` 
Record a video of a trained policy acting in an environment.
```
python run.py enjoy MovingObstaclesNoRules-v0 --algo <default:ppo> --agent path\to\agent.pkl
``` 
Evaluate a trained agent in an environment.
```
python run.py test MovingObstaclesNoRules-v0 --algo ppo --agent path\to\agent.pkl --episodes 1
``` 
