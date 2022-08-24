from basic_model.plant_model import PlantModel

plant = PlantModel()

plant.rotate_node(0, 20)
plant.rotate_node(1, 20)
plant.rotate_node(4, 20)
points = plant.calculate_occlusion(10)

plant.plot_plant(10, points)

print('Done')