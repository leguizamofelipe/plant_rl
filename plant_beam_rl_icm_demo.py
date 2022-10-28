from ICM_Plant.icm import ICM
from basic_model.plant_beam_model_continuous_env import PlantBeamModelContinuousEnvironment

env = PlantBeamModelContinuousEnvironment()

icm = ICM(env=env, beta=0.01, lmd=0.99)
icm.batch_train()
