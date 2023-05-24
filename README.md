# Quantum state_preparation of gravitational waves

A quantum circuit capable of amplitude encoding gravitational wave inspiral waveforms using quantum arithmetic and hybrid quantum-classical machine learning methods.

## Installation

``` pip3 install qiskit matplotlib ```

## Reproducing results

To train the parameterised quantum circuit of 12 layers using the generative adversarial network run:

``` python QGAN_Af.py --nepochs 1450 --reps 12 ```

And similarly for 20 layers:

``` python QGAN_Af.py --nepochs 2260 --reps 20 ```

Simulating the amplitude preparation step using the Grover-Rudolph algorithm is done through:

``` python Af_GR_sim.py ```

And given the weights from the 20 layer trained parameterised quantum circuit is stored in `./weights/weights_20_2260.npy', the amplitude preparation step can be simulated using:

``` python Af_PQC_sim.py ```

The evaluation of the frequency dependent phase is simulated through:

``` python Psif_sim.py ```

The full waveform produced using the Grover-Rudolph algorithm is simulated with:

``` python waveform_GR_sim.py ```

and for the trained parameterised quantum circuit with weights stored in `./weights/weights_20_2260.npy':

``` python waveform_PQC_sim.py ```
