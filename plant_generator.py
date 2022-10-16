from basic_model.plant_beam_model import PlantBeamModel
import random

for i in range(0, 1):
    P = PlantBeamModel(fruit_radius=0)
    P.apply_force(random.random()*400-200, 1.5)
    # P.plot_plant(save = True, filename = f'plants_random/{int(max(P.max_von_mises))}.png')