import math

onsite = 0.0#onsite energy in the scattering region

onsite_l = 0.0  #onsite energy in the left lead

onsite_r = 0.0  #onsite energy in the right lead
    
hopping = -0.58  #the hopping the z direction of the scattering region

hopping_y = -0.58 #the hopping the y direction of the scattering region

hopping_x = -0.58  #the hopping in the x direction of the scattering region

hopping_lz = -0.58  #the hopping in the z direction of the left lead

hopping_ly = -0.58  #the hopping in the y direction of the left lead

hopping_lx = -0.58  #the hopping in the x direction of the left lead

hopping_rz = -0.58  #the hopping in the z direction of the right lead

hopping_ry= -0.58  # the hopping in the y direction of the right lead

hopping_rx = -0.58  #the hopping in the x direction of the left lead

hopping_lc = -0.58 # the hopping inbetween the left lead and scattering region

hopping_rc = -0.58# the hopping inbetween the right lead and scattering region

chain_length = 1 # the number of atoms in the z direction of the scattering region

chain_length_y = 1 # this is the number of k in the y direction for the scattering region

chain_length_x = 1 #This is the number of points in the x direction.

chemical_potential = 0.0

temperature = 0

steps = 161 #number of energy points we take   

e_upper_bound = 20.0 # this is the max energy value

e_lower_bound = -20.0# this is the min energy value
hubbard_interaction = 0.0 # this is the hubbard interaction

voltage_r = [-0.15 * i for i in range(41)]

voltage_l = [0.15 * i for i in range(41)]

voltage_step = 0 # voltage step of zero is equilibrium. This is an integer and higher values correspond to a higher potential difference between the two leads.

pi = 3.14159265359

if (hubbard_interaction == 0.0):
    interaction_order = 0 # this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2#
else:
    interaction_order = 2
#this needs a tiny imaginary part for convergence in the calculation of the embedding self energy
energy = [e_lower_bound+( e_upper_bound - e_lower_bound ) / steps * x +0.00000000001 * 1j for x in range(steps)]

def conjugate(x):
    a = x.real
    b = x.imag
    y = a - 1j * b
    return y

def fermi_function( energy_: complex ):
    if(temperature == 0):
        if( energy_.real < chemical_potential ):
            return 1
        else:
            return 0
    else:
        return 1 / (1 + math.exp( ( energy_.real - chemical_potential ) / temperature ))
