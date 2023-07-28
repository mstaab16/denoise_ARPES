import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import time

def sph_to_xyz(sph_coords):
    def single_coord(coord):
        theta, phi = coord
        return (
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(theta)
                )
    return np.array(list(map(lambda x: single_coord(x), sph_coords)))

def xyz_to_sph(xyz_coords):
    def single_coord(coord):
        x, y, z = coord
        # print(x,y,z)
        # print(np.arccos(x/np.sqrt(x**2 + y**2 + 1e-6)))
        return (
                    np.arccos(z + 1e-10),
                    np.sign(y + 1.12941e-10)*np.arccos(x/np.sqrt(x**2 + y**2 + 1e-10))
                )
    return np.array(list(map(lambda x: single_coord(x), xyz_coords)))

def rotate_z(sph_coords, phi):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1],
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def rotate_y(sph_coords, theta):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def rotate_x(sph_coords, theta):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [1,0,0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles, polar, azimuthal, tilt, angles_in_degrees = True):
    # All angle arguments are in degrees. To use radians use angles_in_degrees = False
    # Zero polar, azimuthal, and tilt corresponds to coordinates along the ky=0 plane (phi = 0) (physics convention)
    # Rotations are applied in this order: 
    #   - Azimuthal (rotate about kz axis)
    #   - Polar_y (rotate about ky axis)
    #   - Polar_x (rotate about kx axis)

    if angles_in_degrees:
        acceptance_angle = np.radians(acceptance_angle)
        polar = np.radians(polar)
        azimuthal = np.radians(azimuthal)
        tilt = np.radians(tilt)
    
    slit_coords = np.array([np.linspace(-acceptance_angle/2, acceptance_angle/2, num_angles), np.zeros(num_angles)]).T

    rotated_coords = sph_to_xyz(rotate_x(rotate_y(rotate_z(slit_coords, azimuthal), tilt), polar))

    kx, ky, kz = rotated_coords.T

    return kx, ky, kz

def fermi_function(temperature, energy):
    """
    This function returns the Fermi function for a given energy and temperature
    """
    beta = 1/(8.617e-5*temperature) #eV^-1
    return 1/(np.exp(beta*(energy)) + 1)

def detector_grid(shape):
    x = np.linspace(-3.14,3.14,shape[1])
    y = np.linspace(-3.14,3.14,shape[0])
    xx, yy = np.meshgrid(x,y)
    # zz = np.sin(15*(np.sin(np.degrees(45))*xx + np.sin(np.degrees(45))*yy))
    # zz += np.sin(15*(np.sin(np.degrees(45))*xx + np.cos(np.degrees(45))*yy))
    freq = 30
    cutoff = 0.7
    zz1 = np.sin((xx + yy)*freq)
    zz2 = np.sin((xx - yy)*freq)
    zz1 = np.where(zz1 > cutoff, 1,0)
    zz2 = np.where(zz2 > cutoff, 1,0)
    zz = np.where(zz1 + zz2 > .5, 1,0)
    # zz = np.where(zz1 > 0.5 or zz2 > 0.5, 1,0)
    return zz

def lorentzian(x, x0, gamma):
    return (1/np.pi)*(gamma/((x - x0)**2 + (gamma/2)**2))


def tight_binding(kx, ky, energies, intrinsic_broadening=0.005, num_bands=None, realistic_cuprate=False):
    if num_bands is None:
        num_bands = np.random.randint(1,10)
    t0 = np.random.uniform(0.001, 0.5, num_bands)
    t1 = np.random.uniform(-0.9, -0.5, num_bands)
    t2 = np.random.uniform(0.01, 0.2, num_bands)
    t3 = np.random.uniform(-0.4, -0.1, num_bands)
    t4 = np.random.uniform(-0.01, -0.1, num_bands)
    t5 = np.random.uniform(0.05, 0.09, num_bands)
    scaling_factors = np.random.uniform(.1, 1, num_bands)

    if realistic_cuprate:
        t0 = [0.090555]
        t1 = [-0.724474]
        t2 = [0.0627473]
        t3 = [-0.165944]
        t4 = [-0.0150311]
        t5 = [0.0731361]
        num_bands = 1


    energy_func = lambda t0, t1, t2, t3, t4, t5, kx, ky: t0 + 0.5 * t1 * ( np.cos(kx) + np.cos(ky)) + t2 * np.cos(kx) * np.cos(ky) + 0.5* t3 * (np.cos(2 * kx) + np.cos(2*ky)) + 0.5 * t4 * (np.cos(2 * kx) * np.cos(ky) + np.cos(kx) * np.cos(2*ky)) + t5 * np.cos(2*kx) * np.cos(2*ky)

    # kx_, ky_, energies = np.meshgrid(kx, ky, energies)
    energies_ = np.tile(energies, (len(kx),1))
    intensity = np.zeros((len(energies), len(kx)))
    for t0_, t1_, t2_, t3_, t4_, t5_, a in zip(t0, t1, t2, t3, t4, t5, scaling_factors):
        energy_locations = energy_func(t0_, t1_, t2_, t3_, t4_, t5_, kx, ky)
        # print(energy_locations)
        intensity_ = lorentzian(energies_.T, energy_locations, intrinsic_broadening)
        # print(intensity_.shape)
        intensity += a * intensity_
    # print(energy_locations.shape)
    # print(intensity.shape)
    return intensity.T / intensity.max()

def pure_spectral_weight(kx, ky, temperature = 30,
                         k_resolution = 0.011, e_resolution = 0.025,
                         energy_range = (-0.7, 0.1), num_energies = 200,
                         noise_level=0.3, lorentzian_width=0.005, random_bands=False):
    """
    This function computes the APRES intesnity after meshing the kx and ky.
    Returns a 3D array of the spectral weight at each kx, ky and energy.
    The spectral weight has the fermi function applied.
    The size of the gaussian convolution is determined by the e and k resolutions.
    The k and e units are in inverse Angstroms and eV respectively.
    """
    

    low_energy = energy_range[0] - 25*(energy_range[1] - energy_range[0])/num_energies
    high_energy = energy_range[1] + 25*(energy_range[1] - energy_range[0])/num_energies
    energies = np.linspace(low_energy, high_energy, num_energies+50)
    # spectral_weight = np.abs(np.random.normal(0,0.005, size=(len(kx), len(energies))).astype(np.float32))
    spectral_weight = np.zeros((len(kx), len(energies)), dtype=np.float32)

    energy_bands = tight_binding(kx, ky, energies, intrinsic_broadening=lorentzian_width, realistic_cuprate=not random_bands)

    # Set spectral weight to 1 where the energy is for a given kx, ky
    # for energy in energy_bands:
    #     for i in range(len(kx)):
    #         for j in range(len(energies)-1):
    #             intensity = (1/2*np.pi)*(lorentzian_width/((energies[j] - energy[i])**2 + (lorentzian_width/2)**2))
    #             # if intensity > 0 and intensity < 1:
    #             spectral_weight[i][j] += intensity
    #             # #if energy[i] is within the energy range pixel, set the spectral weight to 1
    #             # if energy[i] > energies[j] and energy[i] < energies[j+1]:
    #             #     spectral_weight[i][j] += 
    spectral_weight = energy_bands
    
    # calculate the fermi function in the shape of spectral weight
    fermi = fermi_function(temperature, energies)
    fermi = np.tile(fermi, (len(kx),1))
    # multiply the spectral weight by the fermi function
    # spectral_weight = gaussian_filter(spectral_weight, sigma = 0)
    spectral_weight = spectral_weight
    
    raw_spectrum = spectral_weight.copy()
    #spectral_weight *= fermi_function(energies[j])

    # if K_RES and E_RES are not 0, use scipy gaussian_filter to apply resolution blur
    k_width = np.sqrt((kx[-1] - kx[0])**2 + (ky[-1] - ky[0])**2)
    k_resolution_for_scipy = (k_resolution/(k_width))*len(kx)/10
    e_resolution_for_scipy = (e_resolution/(np.max(energy_range) - np.min(energy_range)))*num_energies
    if k_resolution != 0 or e_resolution != 0:
        arpes_sim = gaussian_filter(spectral_weight, sigma = (k_resolution_for_scipy, e_resolution_for_scipy)) 
        high_stats_spectrum = arpes_sim.copy() * fermi
        fermi_noise = fermi * np.random.normal(1, 0.1, size=arpes_sim.shape) * arpes_sim.max()
        scattered_signal = gaussian_filter(raw_spectrum, sigma = (100*k_resolution_for_scipy, 10*e_resolution_for_scipy))
        scattered_signal = (scattered_signal * fermi_noise)
        scattered_signal -= scattered_signal.min()
        scattered_signal /= scattered_signal.max()
        scattered_signal *= noise_level + .1
        arpes_sim *= np.random.uniform(1-noise_level, 1, size=arpes_sim.shape)
        


    # arpes_sim -= arpes_sim.min()
    # arpes_sim /= arpes_sim.max()
    if k_resolution != 0 or e_resolution != 0:
        arpes_sim += scattered_signal

    
    arpes_sim += scattered_signal * detector_grid(arpes_sim.shape) * noise_level
    arpes_sim *= fermi
    arpes_sim += np.random.exponential(noise_level/10, size=arpes_sim.shape)

    
    # high_stats_spectrum *= fermi
    raw_spectrum *= fermi

    return arpes_sim.transpose()/arpes_sim.max(), high_stats_spectrum.transpose()/high_stats_spectrum.max(), raw_spectrum.transpose()/raw_spectrum.max()

def simulate_ARPES_measurement(polar=0.0, tilt=0.0, azimuthal=0.0,
                               photon_energy=100.0, noise_level=0.3,
                               acceptance_angle=30.0, num_angles=250,
                               num_energies=200, temperature=30.0,
                               k_resolution=0.011, e_resolution=0.025,
                               energy_range=(-0.7, 0.1), random_bands=False):
    """
    This function simulates an ARPES measurement for a given set of parameters.
    Returns the simulated ARPES spectrum.
    """
    inverse_hbar_times_sqrt_2me = 0.512316722 #eV^-1/2 Angstrom^-1
    r = inverse_hbar_times_sqrt_2me*np.sqrt(photon_energy) #(1/hbar) * sqrt(2 m_e KE)
    kx, ky, kz = get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles+50, polar, azimuthal, tilt)
    kx *= r
    ky *= r
    kz *= r
    arpes, high_stats, raw_spectrum = pure_spectral_weight(kx, ky, temperature=temperature, k_resolution=k_resolution, e_resolution=e_resolution, energy_range=energy_range, num_energies=num_energies, noise_level=noise_level, random_bands=random_bands)
    return arpes[25:-25,25:-25], high_stats[25:-25,25:-25],raw_spectrum[25:-25,25:-25]


def generate_batch(n, noise=None, k_resolution=None, e_resolution=None, num_angles=256, num_energies=256):
    print("Generating data...")
    X = np.zeros((n, num_angles, num_energies), dtype=np.float32)
    Y = np.zeros((n, num_angles, num_energies), dtype=np.float32)
    
    polar = np.random.uniform(-15,15, n)
    tilt = np.random.uniform(-15,15, n)
    azimuthal = np.random.uniform(0, 360, n)
    photon_energy = np.random.normal(100, 10, n)

    if noise is None:
        noise_level = np.abs(np.random.normal(0.3, 0.1, n))
    else:
        noise_level = np.abs(np.random.normal(noise, noise*0.1, n))
    if k_resolution is None:
        k_resolution = np.random.normal(0.0011, 0.003, n)
    else:
        k_resolution = np.random.normal(k_resolution, k_resolution*0.1, n)
    if e_resolution is None:
        e_resolution = np.random.normal(0.015, 0.003, n)
    else:
        e_resolution = np.random.normal(e_resolution, e_resolution*0.1, n)


    for i in range(n):
        print(f"Generating spectrum {i+1}/{n}", end='\r')
        X_, Y_, _ = simulate_ARPES_measurement(
                                        polar=polar[i], azimuthal=azimuthal[i], tilt=tilt[i],
                                        k_resolution=k_resolution[i], e_resolution=e_resolution[i],
                                        noise_level=noise_level[i], photon_energy=photon_energy[i],
                                        num_angles=256, num_energies=256,
                                        random_bands=True
                                        )
        # X[i] = resize(X_, (64,64))
        # Y[i] = resize(Y_, (64,64))
        X[i] = X_
        Y[i] = Y_
    print(" "*100)
    return X, Y


if __name__ == "__main__":
    # sender = imagezmq.ImageSender(connect_to='tcp://localhost:5432')
    # while True:
    #     azimuth = np.random.uniform(0, 360)
    #     spectrum = simulate_ARPES_measurement(polar=np.random.uniform(-15,15), tilt=np.random.uniform(-15,15), azimuthal=azimuth)
    #     sender.send_image(f"{azimuth:.2f}", spectrum)
    #     time.sleep(0.01)

    # from time import perf_counter_ns
    # test_num = 1
    # start = perf_counter_ns()
    # for _ in range(test_num):
    #     polar = np.random.uniform(-15,15)
    #     tilt = np.random.uniform(-15,15)
    #     azimuthal = np.random.uniform(0, 360)
    #     # k_resolution = np.random.uniform(0.001, 0.02)
    #     # e_resolution = np.random.uniform(0.001, 0.03)
    #     # noise_level = np.random.uniform(0,1)
    #     # photon_energy = np.random.uniform(6, 100)
    #     k_resolution = np.random.normal(0.0011, 0.003)
    #     e_resolution = np.random.normal(0.015, 0.003)
    #     noise_level = np.abs(np.random.normal(0.3, 0.1))
    #     photon_energy = np.random.normal(100, 10)
    #     measured_spectrum, high_stats_spectrum, raw_spectrum = simulate_ARPES_measurement(
    #                                                 polar=polar, azimuthal=azimuthal, tilt=tilt, 
    #                                                 k_resolution=k_resolution, e_resolution=e_resolution,
    #                                                 noise_level=noise_level, photon_energy=photon_energy,
    #                                                 num_angles=256, num_energies=256,
    #                                                 random_bands=True)
    # print(f"Time per spectrum: {(perf_counter_ns() - start)/test_num * 1e-6 :.2f} ms")
    # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    # from skimage.transform import resize
    # measured_spectrum = resize(measured_spectrum, (64,64))
    # high_stats_spectrum = resize(high_stats_spectrum, (64,64))
    # raw_spectrum = resize(raw_spectrum, (64,64))

    # ax1.imshow(measured_spectrum, cmap='Greys', aspect='auto', origin='lower')
    # ax2.imshow(high_stats_spectrum, cmap='Greys', aspect='auto', origin='lower')
    # ax3.imshow(raw_spectrum, cmap='Greys', aspect='auto', origin='lower')
    # plt.show()

    X, Y = generate_batch(1)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.imshow(X[0], cmap='Greys', aspect='auto', origin='lower')
    ax2.imshow(Y[0], cmap='Greys', aspect='auto', origin='lower')
    plt.show()

    # make_training_data()