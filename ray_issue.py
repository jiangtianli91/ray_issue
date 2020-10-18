from src.network import Network
import ray
import pdb

@ray.remote
class ray_network:
    def __init__(self, network):
        self.network = network

    def get_groups(self):
        return self.network.groups

    def get_name(self):
        return self.network.name



ray.init()
xor_net = Network(name="xor")
xor_net.add_group(name="first", num_units=2, group_type="input", input_transforms=[], output_transforms=[])
xor_net.add_group(name="second", num_units=2, group_type="hidden", input_transforms=["dot"], output_transforms=["sigmoid"])
xor_net.add_group(name="third", num_units=1, group_type="output", input_transforms=["dot"], output_transforms=["sigmoid"])
print(xor_net.name)
print(xor_net.groups)

ID = ray_network.remote(xor_net)
print(ray.get(ID.get_name.remote()))
print(ray.get(ID.get_groups.remote()))



