from basic_model.plant_model import PlantModel

plant = PlantModel(10, 5, 5)

# plant.rotate_node(0, 20)
plant.rotate_node(1, 20)
plant.rotate_node(2, 20)
plant.rotate_node(3, 20)
o_factor = plant.calculate_occlusion()

print(f'O factor: {o_factor}')

plant.plot_plant()

print('Done')