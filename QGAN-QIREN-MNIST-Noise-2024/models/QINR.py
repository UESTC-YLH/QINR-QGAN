import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from jax import numpy as np
# import jax
# import os
# os.environ["OMP_NUM_THREADS"] = '10'
image_shape = 28
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit_aer import AerSimulator
p_reset = 0.01
p_meas = 0.15
p_gate1 = 0.02

# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])



class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True, idx=0):
        super().__init__()
        self.idx = idx
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)

    def forward(self, x):
        x1 = self.clayer(x)
        # a = torch.unsqueeze(x1, dim=2)
        # x1 = self.norm(a.permute(0, 2, 1)).permute(0, 2, 1)
        # x1 = x1.squeeze(dim=2)
        out = self.qlayer(x1)
        return out


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layer, use_noise):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise

        def _circuit(inputs, weights1, weights2):
            # noise_circuit = QuantumCircuit(self.in_features)
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)

                # for k in range(self.in_features):
                #     # a = weights1[i][0][k]
                #     noise_circuit.u(*weights1.cpu().detach().numpy()[i][0][k], qubit=k)
                # for j in range(self.in_features):
                #     if j<self.in_features-1:
                #         noise_circuit.cx(j, j + 1)
                #     else:
                #         noise_circuit.cx(self.in_features-1, 0)
                # for k in range(self.in_features):
                #     noise_circuit.u(*weights1.cpu().detach().numpy()[i][1][k], qubit=k)
                # for c in range(0, 4):
                #     noise_circuit.cx(c, c+2)
                # for d in reversed(range(2, 6)):
                #     noise_circuit.cx(d, d-2)

                for j in range(self.in_features):
                    qml.RZ(inputs[0, j], wires=j)
                    # noise_circuit.rz(inputs.cpu().detach().numpy()[0, j], qubit=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)
            # for k in range(self.in_features):
            #     # a = weights1[i][0][k]
            #     noise_circuit.u(*weights2.cpu().detach().numpy()[0][k], qubit=k)
            # for j in range(self.in_features):
            #     if j <self.in_features - 1:
            #         noise_circuit.cx(j, j + 1)
            #     else:
            #         noise_circuit.cx(self.in_features-1, 0)
            # for k in range(self.in_features):
            #     noise_circuit.u(*weights2.cpu().detach().numpy()[1][k], qubit=k)
            # for c in range(0, 4):
            #     noise_circuit.cx(c, c + 2)
            #
            # for d in reversed(range(2, 6)):
            #     noise_circuit.cx(d, d - 2)
            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)
            # noise_circuit.measure_all()
            # sim_noise = AerSimulator(noise_model=noise_bit_flip)
            # circ_to_noise = transpile(noise_circuit, sim_noise)
            # result_bit_flip = sim_noise.run(circ_to_noise).result()
            # counts_bit_flip = result_bit_flip.get_counts(0)
            # return
            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            # res
            return res


        torch_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3), "weights2": (2, in_features, 3)}

        # self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")
        self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")

        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        res = self.qnn(x)

        # print(result_bit_flip)
        return res.reshape(orgin_shape)

class PQWGAN_CC():
    def __init__(self, image_size, channels, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True):
        self.image_shape = (channels, image_size, image_size)
        self.critic = self.ClassicalCritic(self.image_shape)
        self.generator = self.Hybridren(in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True)

    class ClassicalCritic(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape
            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)

    class Hybridren(nn.Module):
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True):
            #n_layers=hidden_layers
            super().__init__()

            self.net = []
            self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1))

            for i in range(hidden_layers):
                self.net.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))

            if outermost_linear:
                final_linear = nn.Linear(hidden_features, 128)

            else:
                final_linear = HybridLayer(hidden_features, out_features, spectrum_layer, use_noise)

            final_linear_1 = nn.Linear(128, 512)
            final_linear_2 = nn.Linear(512, 256)
            final_linear_3 = nn.Linear(256, int(np.prod(image_shape*image_shape)))
            self.net.append(final_linear)
            self.net.append(final_linear_1)
            self.net.append(final_linear_2)
            self.net.append(final_linear_3)
            self.net = nn.Sequential(*self.net)

        def forward(self, coords):
            coords = coords.clone().detach().requires_grad_(True)
            output = self.net(coords)
            final_out_new = output.view(output.shape[0], 1, image_shape, image_shape)
            # coords是原来的输入 -1到1之间的1000个采样点  output是生成数据
            return final_out_new




if __name__ == "__main__":
    gen = PQWGAN_CC(image_size=16, channels=1, n_generators=16, n_qubits=5, n_ancillas=1, n_layers=1).generator
    print(qml.draw(gen.qnode)(torch.rand(5), torch.rand(1, 5, 3)))