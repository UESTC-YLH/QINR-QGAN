import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
image_shape = 28

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
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)
                for j in range(self.in_features):
                    qml.RZ(inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        torch_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3), "weights2": (2, in_features, 3)}

        self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")

        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        out = self.qnn(x)
        return out.reshape(orgin_shape)

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

    # class QuantumGenerator(nn.Module):
    #     def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, image_shape, patch_shape):
    #         super().__init__()
    #         self.n_generators = n_generators
    #         self.n_qubits = n_qubits
    #         self.n_ancillas = n_ancillas
    #         self.n_layers = n_layers
    #         self.q_device = qml.device("default.qubit", wires=n_qubits)
    #         self.params = nn.ParameterList([nn.Parameter(torch.rand(n_layers, n_qubits, 3), requires_grad=True) for _ in range(n_generators)])
    #         self.qnode = qml.QNode(self.circuit, self.q_device, interface="torch")
    #         self.image_shape = image_shape
    #         self.patch_shape = patch_shape
    #
    #         self.gfc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
    #         self.gfc2 = nn.Linear(512, 256)
    #         self.gfc3 = nn.Linear(256, int(np.prod(self.image_shape)))
    #
    #     def forward(self, x):
    #         special_shape = bool(self.patch_shape[0]) and bool(self.patch_shape[1])
    #         patch_size = 2 ** (self.n_qubits - self.n_ancillas)
    #         image_pixels = self.image_shape[2] ** 2
    #         pixels_per_patch = image_pixels // self.n_generators
    #         if special_shape and self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
    #             raise ValueError("patch shape and patch size dont match!")
    #         output_images = torch.Tensor(x.size(0), 0)
    #
    #         for sub_generator_param in self.params:
    #             patches = torch.Tensor(0, pixels_per_patch)
    #             for item in x:
    #                 sub_generator_out = self.partial_trace_and_postprocess(item, sub_generator_param).float().unsqueeze(0)
    #                 if pixels_per_patch < patch_size:
    #                     sub_generator_out = sub_generator_out[:, :pixels_per_patch]
    #                 patches = torch.cat((patches, sub_generator_out))
    #             output_images = torch.cat((output_images, patches), 1)
    #
    #         if special_shape:
    #             final_out = torch.zeros(x.size(0), *self.image_shape)
    #             for i, img in enumerate(output_images):
    #                 for patches_done, j in enumerate(range(0, img.shape[0], pixels_per_patch)):
    #                     patch = torch.reshape(img[j:j + pixels_per_patch], self.patch_shape)
    #                     starting_h = ((patches_done * self.patch_shape[1]) // self.image_shape[2]) * self.patch_shape[0]
    #                     starting_w = (patches_done * self.patch_shape[1]) % self.image_shape[2]
    #                     final_out[i, 0, starting_h:starting_h + self.patch_shape[0],
    #                     starting_w:starting_w + self.patch_shape[1]] = patch
    #         else:
    #             final_out = output_images.view(output_images.shape[0], *self.image_shape)
    #         final_out = final_out.view(final_out.shape[0], -1)
    #         x = self.gfc1(final_out)
    #         x = self.gfc2(x)
    #         final_out = self.gfc3(x)
    #         final_out_new = final_out.view(final_out.shape[0], *self.image_shape)
    #         return final_out_new

        # def circuit(self, latent_vector, weights):
        #     for i in range(self.n_qubits):
        #         qml.RY(latent_vector[i], wires=i)
        #
        #     for i in range(self.n_layers):
        #         for j in range(self.n_qubits):
        #             qml.Rot(*weights[i][j], wires=j)
        #
        #         for j in range(self.n_qubits - 1):
        #             qml.CNOT(wires=[j, j + 1])
        #
        #     return qml.probs(wires=list(range(self.n_qubits)))
        #
        # def partial_trace_and_postprocess(self, latent_vector, weights):
        #     probs = self.qnode(latent_vector, weights)
        #     probs_given_ancilla_0 = probs[:2 ** (self.n_qubits - self.n_ancillas)]
        #     post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)
        #
        #     # uncomment to check outputs of circuit
        #     # print(torch.sum(post_measurement_probs[:28]), torch.sum(post_measurement_probs[28:]))
        #
        #     # normalise image between [-1, 1]
        #     post_processed_patch = ((post_measurement_probs / torch.max(post_measurement_probs)) - 0.5) * 2
        #     return post_processed_patch


if __name__ == "__main__":
    gen = PQWGAN_CC(image_size=16, channels=1, n_generators=16, n_qubits=5, n_ancillas=1, n_layers=1).generator
    print(qml.draw(gen.qnode)(torch.rand(5), torch.rand(1, 5, 3)))