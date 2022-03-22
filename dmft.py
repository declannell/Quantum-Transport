import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import time
import leads_self_energy
import noninteracting_gf
import parameters
import warnings
from typing import List


class Interacting_GF:
    kx: float
    ky: float
    voltage_step: float
    hamiltonian: List[List[complex]]
    effective_hamiltonian: List[List[List[complex]]]
    interacting_gf: List[List[List[complex]]]
    self_energy_many_body: List[List[List[complex]]]

    def __init__(self, _kx: float, _ky: float, _voltage_step: int, _self_energy_many_body):
        self.kx = _kx
        self.ky = _ky
        self.voltage_step = _voltage_step
        self.self_energy_many_body = _self_energy_many_body
        self.hamiltonian = create_matrix(parameters.chain_length)
        self.effective_hamiltonian = [create_matrix(
            parameters.chain_length) for r in range(parameters.steps)]
        self.interacting_gf = [create_matrix(
            parameters.chain_length) for r in range(parameters.steps)]
        # this willgetting the embedding self energies from the leads code
        self.get_effective_matrix()
        self.get_interacting_gf()

    def get_effective_matrix(self):
        self_energy = leads_self_energy.EmbeddingSelfEnergy(
            self.kx, self.ky, parameters.voltage_step)
        # self_energy.plot_self_energy()
        for i in range(0, parameters.chain_length-1):
            self.hamiltonian[i][i+1] = parameters.hopping
            self.hamiltonian[i+1][i] = parameters.hopping
        for i in range(0, parameters.chain_length):
            voltage_i = parameters.voltage_l[self.voltage_step] - (i + 1) / (float)(parameters.chain_length + 1) * (
                parameters.voltage_l[self.voltage_step] - parameters.voltage_r[self.voltage_step])
            #print("The external voltage is on site ",  i, " is ", voltage_i)
            self.hamiltonian[i][i] = parameters.onsite + 2 * parameters.hopping_x * \
                math.cos(self.kx) + 2 * parameters.hopping_y * \
                math.cos(self.ky) + voltage_i
            for j in range(0, parameters.chain_length):
                for r in range(0, parameters.steps):
                    self.effective_hamiltonian[r][i][j] = self.hamiltonian[i][j]

        for r in range(0, parameters.steps):
            self.effective_hamiltonian[r][0][0] += self_energy.self_energy_left[r]
            self.effective_hamiltonian[r][-1][-1] += self_energy.self_energy_right[r]
        """    
        plt.plot(parameters.energy, [e[0][0].real for e in self.effective_hamiltonian], color='red', label='real effective hamiltonian') 
        plt.plot(parameters.energy, [e[0][0].imag for e in self.effective_hamiltonian], color='blue', label='Imaginary effective hamiltonian')
        plt.title("effective hamiltonian")
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        plt.ylabel("effective hamiltonian")  
        plt.show()   
        """

    def get_interacting_gf(self):
        inverse_green_function = create_matrix(parameters.chain_length)
        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                for j in range(0, parameters.chain_length):
                    if (i == j):
                        inverse_green_function[i][j] = parameters.energy[r].real - \
                            self.effective_hamiltonian[r][i][j] - \
                            self.self_energy_many_body[r][i]
                    else:
                        inverse_green_function[i][j] = - \
                            self.effective_hamiltonian[r][i][j]

            self.interacting_gf[r] = la.inv(
                inverse_green_function, overwrite_a=False, check_finite=True)

    def plot_greenfunction(self):
        for i in range(0, parameters.chain_length):

            plt.plot(parameters.energy, [
                     e[i][i].real for e in self.interacting_gf], color='red', label='Real Green up')
            plt.plot(parameters.energy, [
                     e[i][i].imag for e in self.interacting_gf], color='blue', label='Imaginary Green function')
            j = i + 1
            plt.title('Noninteracting Green function site %i' % j)
            plt.legend(loc='upper left')
            plt.xlabel("energy")
            plt.ylabel("Noninteracting green Function")
            plt.show()

    # this allows me to print the effective hamiltonian if called for a certain energy point specified by num.
    def print_hamiltonian(self):
        # eg. hamiltonian.print(4) will print the effective hamiltonian of the 4th energy step
        for i in range(0, parameters.chain_length):
            # rjust adds padding, join connects them all
            row_string = " ".join((str(r).rjust(5, " ")
                                  for r in self.hamiltonian[i]))
            print(row_string)


# this should work as in first order interaction, it gives the same result as fluctuation dissaption thm to 11 deciaml places
def get_spin_occupation(gf_lesser_up: List[complex], gf_lesser_down: List[complex]):
    delta_energy = (parameters.e_upper_bound -
                    parameters.e_lower_bound)/parameters.steps
    result_up, result_down = 0, 0
    for r in range(0, parameters.steps):
        result_up = (delta_energy) * gf_lesser_up[r] + result_up
        result_down = (delta_energy) * gf_lesser_down[r] + result_down
    x = -1j / (2 * parameters.pi) * result_up
    y = -1j / (2 * parameters.pi) * result_down
    return x, y


def integrate(gf_1: List[complex], gf_2: List[complex], gf_3: List[complex], r: int):
    # in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded).The
    delta_energy = (parameters.e_upper_bound -
                    parameters.e_lower_bound) / parameters.steps
    result = 0
    for i in range(0, parameters.steps):
        for j in range(0, parameters.steps):
            if (((i + j - r) >= 0) and ((i + j - r) < parameters.steps)):
                # this integrates like PHYSICAL REVIEW B 74, 155125 2006
                # I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
                # and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps. These conditions enesure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
                result = (delta_energy / (2 * parameters.pi)) ** 2 * \
                    gf_1[i] * gf_2[j] * gf_3[i+j-r] + result
            else:
                result = result
    return result


# this creates the entire parameters.energy() array at once
def self_energy_2nd_order(impurity_gf_up: List[complex], impurity_gf_down: List[complex], impurity_gf_up_lesser: List[complex], impurity_gf_down_lesser: List[complex]):
    impurity_self_energy = [0 for z in range(0, parameters.steps)]

    # the are calculating the self parameters.energy() sigma_{ii}(E) for each discretized parameters.energy(). To do this we pass the green_fun_{ii} for all energies as we need to integrate over all energies in the integrate function
    for r in range(0, parameters.steps):
        impurity_self_energy[r] = parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up, impurity_gf_down, impurity_gf_down_lesser, r))  # line 3
        impurity_self_energy[r] += parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up, impurity_gf_down_lesser, impurity_gf_down_lesser, r))  # line 2
        impurity_self_energy[r] += parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up_lesser, impurity_gf_down, impurity_gf_down_lesser, r))  # line 1
        impurity_self_energy[r] += parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up_lesser, impurity_gf_down_lesser, [parameters.conjugate(e) for e in impurity_gf_down], r))  # line 4
    return impurity_self_energy

# this is only used to compare the lesser green functions using two different methods. This is not used in the calculation of the self energies.
def fluctuation_dissipation(green_function: List[complex]):
    g_lesser = [0 for z in range(0, parameters.steps)]
    for r in range(0, parameters.steps):
        g_lesser[r] = - parameters.fermi_function(parameters.energy[r].real) * (
            green_function[r] - parameters.conjugate(green_function[r]))
    return g_lesser


def impurity_solver(impurity_gf_up: List[complex], impurity_gf_down: List[complex]):
    impurity_self_energy_up = [0 for z in range(0, parameters.steps)]
    impurity_self_energy_down = [0 for z in range(0, parameters.steps)]

    if (parameters.voltage_step == 0):
        impurity_gf_up_lesser = fluctuation_dissipation(impurity_gf_up)
        impurity_gf_down_lesser = fluctuation_dissipation(impurity_gf_down)

    if (parameters.interaction_order > 0):
        impurity_spin_up, impurity_spin_down = get_spin_occupation(
            impurity_gf_up_lesser, impurity_gf_down_lesser)

    if (parameters.interaction_order == 2):
        impurity_self_energy_up = self_energy_2nd_order(
            impurity_gf_up, impurity_gf_down, impurity_gf_up_lesser, impurity_gf_down_lesser)
        impurity_self_energy_down = self_energy_2nd_order(
            impurity_gf_down, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_up_lesser)
        for r in range(0, parameters.steps):
            impurity_self_energy_up[r] += parameters.hubbard_interaction * impurity_spin_down
            impurity_self_energy_down[r] += parameters.hubbard_interaction * impurity_spin_up

    if (parameters.interaction_order == 1):
        for r in range(0, parameters.steps):
            impurity_self_energy_up[r] = parameters.hubbard_interaction * \
                impurity_spin_down
            impurity_self_energy_down[r] = parameters.hubbard_interaction * \
                impurity_spin_up

    return impurity_self_energy_up, impurity_self_energy_down, impurity_spin_up, impurity_spin_down


def sum_gf_interacting(r, i, j, gf_interacting_up, gf_interacting_down):
    up = 0.0
    down = 0.0
    num_k_points = parameters.chain_length_x * parameters.chain_length_y
    for kx_i in range(0, parameters.chain_length_x):
        for ky_i in range(0, parameters.chain_length_y):
            up += (
                gf_interacting_up[ky_i][kx_i].interacting_gf[r][i][j]
                / num_k_points)
            down += (
                gf_interacting_down[ky_i][kx_i].interacting_gf[r][i][j]
                / num_k_points)
    return (up, down)

def create_matrix(size: int):
    return [[0.0 for x in range(size)] for y in range(size)]

def dmft(voltage: int, kx: List[float], ky: List[float]):
    self_energy_mb_up = [
        [0 for i in range(0, parameters.steps)]for z in range(0, parameters.steps)]
    self_energy_mb_down = [
        [0 for i in range(0, parameters.steps)]for z in range(0, parameters.steps)]

    n = parameters.chain_length**2 * parameters.steps
    differencelist = [0 for i in range(0, 2 * n)]
    old_green_function = [[[1.0 + 1j for x in range(parameters.chain_length)] for y in range(
        parameters.chain_length)] for z in range(0, parameters.steps)]
    difference = 100.0
    count = 0
    # these allows us to determine self consistency in the retarded green function
    while (difference > 0.0001 and count < 15):
        count += 1
        gf_interacting_up = [[Interacting_GF(kx[i], ky[j], voltage, self_energy_mb_up) for i in range(
            0, parameters.chain_length_x)] for j in range(0, parameters.chain_length_y)]
        gf_interacting_down = [[Interacting_GF(kx[i], ky[j], voltage, self_energy_mb_down) for i in range(
            0, parameters.chain_length_x)] for j in range(0, parameters.chain_length_y)]

        # this quantity is the green function which is averaged over all k points.
        gf_local_up = [create_matrix(parameters.chain_length)
                       for z in range(0, parameters.steps)]
        gf_local_down = [create_matrix(parameters.chain_length)
                         for z in range(0, parameters.steps)]

    # for r, i, j in cartesian(parameters.steps, parameters.chain_length, parameters.chain_length):

        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                for j in range(0, parameters.chain_length):
                    (up, down) = sum_gf_interacting(
                        r, i, j, gf_interacting_up, gf_interacting_down)
                    gf_local_up[r][i][j] += up
                    gf_local_down[r][i][j] += down

        # this will compare the new green function with the last green function for convergence
        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                # this is due to the spin_up_occup being of length chain_length
                for j in range(0, parameters.chain_length):
                    differencelist[r + i + j] = abs(
                        gf_local_up[r][i][j].real - old_green_function[r][i][j].real)
                    differencelist[n + r + i + j] = abs(
                        gf_local_up[r][i][j].imag - old_green_function[r][i][j].imag)
                    old_green_function[r][i][j] = gf_local_up[r][i][j]

        difference = max(differencelist)

        if (difference < 0.0001):
            break

        if(parameters.interaction_order != 0):
            for i in range(0, parameters.chain_length):
                impurity_self_energy_up, impurity_self_energy_down, spin_up_occup, spin_down_occup = (
                    impurity_solver([e[i][i] for e in gf_local_up], [e[i][i] for e in gf_local_down]))
                """
                print("The spin up occupancy for the site",
                      i + 1, " is ", spin_up_occup)
                print("The spin down occupancy for the site",
                      i + 1, " is ", spin_down_occup)
                """
                for r in range(0, parameters.steps):
                    self_energy_mb_up[r][i] = impurity_self_energy_up[r]
                    self_energy_mb_down[r][i] = impurity_self_energy_down[r]
        else:
            break
        print("The count is ", count, "The difference is ", difference)

    for i in range(0, parameters.chain_length):
        plt.plot(parameters.energy, [
            e[i][i].real for e in gf_local_up], color='red', label='Real Green up')
        plt.plot(parameters.energy, [
            e[i][i].imag for e in gf_local_up], color='blue', label='Imaginary Green function')
        j = i + 1
        plt.title('The local Green function site %i' % j)
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        plt.ylabel("Noninteracting green Function")
        plt.show()

    for i in range(0, parameters.chain_length):
        fig = plt.figure()
        plt.plot(parameters.energy, [
                 e[i].imag for e in self_energy_mb_down], color='blue', label='imaginary self energy')
        plt.plot(parameters.energy, [
                 e[i].real for e in self_energy_mb_down], color='red', label='real self energy')
        j = i + 1
        plt.title('Many-body self energy spin down site %i' % j)
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")
        plt.show()


    for i in range(0, parameters.chain_length):
        fig = plt.figure()
        plt.plot(parameters.energy, [
                 e[i].imag for e in self_energy_mb_up], color='blue', label='imaginary self energy')
        plt.plot(parameters.energy, [
                 e[i].real for e in self_energy_mb_up], color='red', label='real self energy')
        j = i + 1
        plt.title('Many-body self energy spin up site %i' % j)
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")
        plt.show()

    #print("The spin up occupaton probability is ", spin_up_occup)
    #print("The spin down occupaton probability is ", spin_down_occup)
    # if(voltage == 0):#this compares the two methods in equilibrium
        #compare_g_lesser(gf_int_lesser_up , gf_int_up)
    for i in range(0, parameters.chain_length):
        print("The spin up occupancy for the site",
            i + 1, " is ", spin_up_occup)
        print("The spin down occupancy for the site",
            i + 1, " is ", spin_down_occup)
    print("The count is ", count)
    return gf_local_up, gf_local_down  # , spin_up_occup, spin_down_occup


def main():
    kx = [0 for m in range(0, parameters.chain_length_x)]
    ky = [0 for m in range(0, parameters.chain_length_y)]
    for i in range(0, parameters.chain_length_y):
        if (parameters.chain_length_y != 1):
            ky[i] = 2 * parameters.pi * i / parameters.chain_length_y
        elif (parameters.chain_length_y == 1):
            ky[i] = parameters.pi / 2.0

    for i in range(0, parameters.chain_length_x):
        if (parameters.chain_length_x != 1):
            kx[i] = 2 * parameters.pi * i / parameters.chain_length_x
        elif (parameters.chain_length_x == 1):
            kx[i] = parameters.pi / 2.0

    # voltage step of zero is equilibrium.
    print("The voltage difference is ",
          parameters.voltage_l[parameters.voltage_step] - parameters.voltage_r[parameters.voltage_step])
    print("The number of sites in the z direction is ", parameters.chain_length)
    print("The number of sites in the x direction is ", parameters.chain_length_x)
    print("The number of sites in the y direction is ", parameters.chain_length_y)
    print("The ky value is ", ky)
    print("The kx value is ", kx)

    green_function_up, green_function_down = dmft(
        parameters.voltage_step, kx, ky)


if __name__ == "__main__":  # this will only run if it is a script and not a import module
    main()
