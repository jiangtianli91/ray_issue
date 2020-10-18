import ray
import numpy as np

@ray.remote
class ray_network:
    def __init__(self, network):
        self.network = network

    def get_groups(self):
        return self.network.groups

    def get_name(self):
        return self.network.name


class Group:
    incoming_links = None
    outgoing_links = None

    name = None
    num_units = 0
    group_type = 0
    input_matrix = None
    output_matrix = None
    input_derivs = None
    output_derivs = None
    incoming_derivs = None
    incoming_weights = None


    def __init__(self, name, num_units, group_type, input_transforms, output_transforms, time_intervals, ticks_per_interval):
        self.name = name
        # append 1 to number of units to account for bias unit
        self.num_units = num_units
        self.group_type = group_type
        self.num_cols = 0
        self.unit_names = []

        self.names = [name + str(i) for i in range(0, num_units)]

        # initialize arrays
        self.input_matrix = np.zeros(self.num_units)
        self.output_matrix = np.zeros((1, self.num_units))
        self.input_derivs = np.zeros(self.num_units)
        self.output_derivs = np.zeros((1, self.num_units))

        self.incoming_links = []
        self.outgoing_links = []
        self.incoming_derivs = np.zeros(self.num_units)
        self.incoming_weights = np.array([[1]])




class Network:
    name = None
    time_intervals = 0
    input_groups = []
    output_groups = []
    groups = []
    bias = None


    def __init__(self, name, time_intervals=1, ticks_per_interval=1, learning_rate=0.2, add_bias=True):
        self.name = name
        self.time_intervals = time_intervals
        self.ticks_per_interval = ticks_per_interval
        self.learning_rate = learning_rate
        self.num_groups = 0
        self.num_units = 0


    def add_group(self, name, num_units, group_type, input_transforms, output_transforms):

        # check if name exists
        if self.check_name(name):

            # instantiate a new group object, append it to master list of groups
            new_group = Group(name, num_units, group_type, input_transforms, output_transforms, self.time_intervals,
                              self.ticks_per_interval)
            if group_type != "input" and group_type != "bias" and self.bias is not None:
                new_group.add_bias(self.bias)
            self.num_groups += 1
            self.num_units += num_units

            self.groups.append(new_group)

            # check if the group is of input or output type, and append it to appropriate array
            if group_type == "input":
                self.input_groups.append(new_group)
            elif group_type == "output":
                self.output_groups.append(new_group)
            elif group_type == "bias":
                self.bias = new_group
                self.bias.output_matrix = np.array([1])

    def check_name(self, name):
        result = True

        for group in self.groups:
            if (group.name == name):
                result = False

        return result



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



