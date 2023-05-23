import torch, time, argparse, sys
import numpy as np
from datetime import datetime
from qiskit.utils import algorithm_globals
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch import nn
from torch.optim import Adam
from scipy.stats import multivariate_normal, entropy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
ticksize=22
figsize=(10,10)

def QGAN_Af(nx, fmin=40., fmax=168., rseed=None, reps=20, n_epochs=2000, shots=10000, lr=0.01, b1=0.7, b2=0.999):

    if rseed==None:
        rseed = int(datetime.now().timestamp())

    algorithm_globals.random_seed = rseed
    _ = torch.manual_seed(rseed) 

    df = (fmax-fmin)/(2**nx)

    num_dim = 1
    num_discrete_values = 2**nx
    num_qubits = num_dim * int(np.log2(num_discrete_values))

    coords = np.linspace(-2, 2, num_discrete_values)
    grid_elements = np.expand_dims(coords,axis=1)
    prob_data = (np.arange(fmin, fmax, df)**(-7./3))
    prob_data = prob_data / np.sum(prob_data)

    sampler = Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed})

    qc = QuantumCircuit(num_qubits)
    qc.h(qc.qubits)

    ansatz = RealAmplitudes(num_qubits, reps=reps)
    qc.compose(ansatz, inplace=True)

    def create_generator() -> TorchConnector:
        qnn = SamplerQNN(
            circuit=qc,
            sampler=sampler,
            input_params=[],
            weight_params=qc.parameters,
            sparse=False,
        )

        initial_weights = algorithm_globals.random.random(qc.num_parameters)
        return TorchConnector(qnn, initial_weights)    

    class Discriminator(nn.Module):
        def __init__(self, input_size):
            super(Discriminator, self).__init__()

            self.linear_input = nn.Linear(input_size, 20)
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.linear20 = nn.Linear(20, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = self.linear_input(input)
            x = self.leaky_relu(x)
            x = self.linear20(x)
            x = self.sigmoid(x)
            return x

    def adversarial_loss(input, target, w):
        bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
        weighted_loss = w * bce_loss
        total_loss = -torch.sum(weighted_loss)
        return total_loss

    generator = create_generator()
    discriminator = Discriminator(num_dim)

    generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)

    generator_loss_values = []
    discriminator_loss_values = []
    entropy_values = []

    num_qnn_outputs = num_discrete_values**num_dim

    n_epochs = n_epochs-len(entropy_values)
        
    start = time.time()
    for epoch in range(n_epochs):

        valid = torch.ones(num_qnn_outputs, 1, dtype=torch.float)
        fake = torch.zeros(num_qnn_outputs, 1, dtype=torch.float)

        # Configure input
        real_dist = torch.tensor(prob_data, dtype=torch.float).reshape(-1, 1)

        # Configure samples
        samples = torch.tensor(grid_elements, dtype=torch.float)
        disc_value = discriminator(samples)

        # Generate data
        gen_dist = generator(torch.tensor([])).reshape(-1, 1)

        # Train generator
        generator_optimizer.zero_grad()
        generator_loss = adversarial_loss(disc_value, valid, gen_dist)

        # store for plotting
        generator_loss_values.append(generator_loss.detach().item())

        generator_loss.backward(retain_graph=True)
        generator_optimizer.step()

        # Train Discriminator
        discriminator_optimizer.zero_grad()

        real_loss = adversarial_loss(disc_value, valid, real_dist)
        fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
        discriminator_loss = (real_loss + fake_loss) / 2

        # Store for plotting
        discriminator_loss_values.append(discriminator_loss.detach().item())

        discriminator_loss.backward()
        discriminator_optimizer.step()

        with torch.no_grad():
            generated_weights = generator.weight.detach().numpy().reshape((reps+1,num_qubits))
        
        circ = QuantumCircuit(num_qubits)
        circ.h(circ.qubits)

        for rep in np.arange(reps+1):
            for i in np.arange(num_qubits):
                circ.ry(generated_weights[rep,i],circ.qubits[i])
            for i in np.arange(num_qubits):
                if i!=num_qubits-1 and rep!=reps:
                    circ.cx(circ.qubits[i], circ.qubits[i+1])
        
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector())
        
        entropy_value = 1-np.sqrt(np.abs(np.dot(np.sqrt(prob_data),np.conjugate(state_vector)))**2)
        entropy_values.append(entropy_value)
        
        print(np.round(100.*(epoch/n_epochs),2),'%','Mismatch:',"{:.2e}".format(entropy_value),'Generator loss:', np.round(generator_loss_values[-1],2), 'Discriminator loss:', np.round(discriminator_loss_values[-1],2))
        sys.stdout.write("\033[F")

    print(np.round(100.*(epoch/n_epochs),2),'%','Mismatch:',"{:.2e}".format(entropy_value),'Generator loss:', np.round(generator_loss_values[-1],2), 'Discriminator loss:', np.round(discriminator_loss_values[-1],2))

    elapsed = time.time() - start
    print(f"Fit in {elapsed:0.2f} sec")

    with torch.no_grad():
        generated_weights = generator.weight.detach().numpy().reshape((reps+1,num_qubits))

    np.save('./weights/weights_'+str(int(reps))+'_'+str(int(n_epochs)),generated_weights)
    np.save('./weights/cross_entropy_'+str(int(reps))+'_'+str(int(n_epochs)), entropy_values)
    np.save('./weights/gen_loss_'+str(int(reps))+'_'+str(int(n_epochs)), generator_loss_values)
    np.save('./weights/disc_loss_'+str(int(reps))+'_'+str(int(n_epochs)), discriminator_loss_values)

    circ = QuantumCircuit(num_qubits)
    circ.h(circ.qubits)

    for rep in np.arange(reps+1):
        for i in np.arange(num_qubits):
            circ.ry(generated_weights[rep,i],circ.qubits[i])
        for i in np.arange(num_qubits):
            if i!=num_qubits-1 and rep!=reps:
                circ.cx(circ.qubits[i], circ.qubits[i+1])

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    state_vector = np.asarray(result.get_statevector())

    fidelity = np.abs(np.dot(np.sqrt(prob_data),np.conjugate(state_vector)))**2

    mismatch = 1. - np.sqrt(fidelity)

    print('Fidelity:', fidelity, 'Mismatch:', mismatch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Train QGAN to perform amplitude preparation.")
    parser.add_argument('--nx', help="Number of frequency qubits.", default=6)
    parser.add_argument('--fmin', help="Minimum frequency.", default=40.)
    parser.add_argument('--fmax', help="Maximum frequency.", default=168.)
    parser.add_argument('--b1', help="Adam optimizer b1 parameter.", default=0.7)
    parser.add_argument('--b2', help="Adam optimizer b2 parameter.", default=0.999)
    parser.add_argument('--rseed', help="Random seed.", default=1680458526)
    parser.add_argument('--reps', help="Layers in neural network.", default=20)
    parser.add_argument('--nepochs', help="Number of epochs.", default=2000)
    parser.add_argument('--shots', help="Runs of the neural network to determine output distribution.", default=10000)
    parser.add_argument('--lr', help="Learning rate.", default=0.01)

    opt = parser.parse_args()

    QGAN_Af(opt.nx, fmin=opt.fmin, fmax=opt.fmax, rseed=opt.rseed, reps=opt.reps, n_epochs=opt.nepochs, shots=opt.shots, lr=opt.lr, b1=opt.b1, b2=opt.b2)
