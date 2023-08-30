import PyNAFF as pnf
import numpy as np
import read_dat
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
import json
from resonance_diag import resonance_plot
import naff_shinji

comm =  MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

with open("configs.json", "r") as jsonfile:
    configs = json.load(jsonfile)
filepaths = configs['filepaths']
filenames = configs['filenames']

def probes_tune_calc(probe_list, co_probes, naff=True):
    if rank == 0:
        tune_calc_time1 = MPI.Wtime()
    steps_per_turn = 2850
    naff_r_tunes = []
    naff_z_tunes = []
    fft_r_tunes = []
    fft_z_tunes = []
    lost_particles =[]

    num_particles = max(probe_list[0]["id"]) +1
    print("number of particles = ", num_particles)
    #co_radial_vals = np.array([])
    #z_co_coord_vals = np.array([])
    #for i in range(10):
    #    co_radial_vals = np.concatenate((co_radial_vals, np.array(list(co_dict.values()))))
    #    z_co_coord_vals =np.concatenate((z_co_coord_vals, np.array(list(z_co_dict.values()))))

    co_probe_alldata_df = pd.concat(co_probes)
    co_probe_alldata_df.sort_values(by=["time (s)"], inplace=True)


    # concat all the probes together and order by time
    probe_alldata_df = pd.concat(probe_list)
    probe_alldata_df.sort_values(by=["time (s)"], inplace=True)

    num_per_rank = num_particles // size
    lower_bound =  rank * num_per_rank
    upper_bound = (rank + 1) * num_per_rank

    #select particle from df
    co_x_data = co_probe_alldata_df['# x (m)'].to_numpy()
    co_y_data = co_probe_alldata_df['y (m)'].to_numpy()
    co_z_data = co_probe_alldata_df[' z (m)'].to_numpy()

    comm.Barrier()

    #times = probe_alldata_df['time (s)'].to_list()
    #print(probe_alldata_df['time (s)'])
    #time0 = times[-2]

    #cell_to_cell_time =times[-1] - time0

    #print(probe_list)
    probe1 = probe_list[0]
    probe1.sort_values(by=["time (s)"], inplace=True)
    probe1_particle1 = probe1[probe1["id"] == 0]
    probe1_particle1 = probe1_particle1['time (s)'].to_list()
    #print(probe1['time (s)'])

    probe2 = probe_list[1]
    probe2.sort_values(by=["time (s)"], inplace=True)
    probe2_particle1 = probe2[probe2["id"] == 0]
    probe2_particle1 = probe2_particle1['time (s)'].to_list()
    #print(probe1['time (s)'])

    tof = probe1_particle1[4] - probe1_particle1[3]
    print("time of flight = ", tof)

    cell_to_cell_time = probe2_particle1[3] - probe1_particle1[3]
    print("cell_to_cell_time ", cell_to_cell_time)


    if num_particles !=1:
        lower_bound= lower_bound+1
    for i in range(lower_bound, upper_bound): 
         # re organise data into 1 particles passes through all 16 probes        
        particle_df = probe_alldata_df[probe_alldata_df["id"] == i]
        #print(particle_df)
        #select particle from df
        x_data = particle_df['# x (m)'].to_numpy()
        y_data = particle_df['y (m)'].to_numpy()
        z_data = particle_df[' z (m)'].to_numpy()

        if len(x_data) >= len(co_x_data):
            radial  = np.sqrt(x_data**2 + y_data**2)[:len(co_x_data)] - np.sqrt(co_x_data**2 + co_y_data**2)
            zcoords_diff = z_data[:len(co_z_data)] - co_z_data

        elif len(x_data) <= len(co_x_data):
            radial  = np.sqrt(x_data**2 + y_data**2) - np.sqrt(co_x_data**2 + co_y_data**2)[:len(x_data)]
            zcoords_diff = z_data - co_z_data[:len(z_data)]

        elif len(x_data) == len(co_x_data):
            radial  = np.sqrt(x_data**2 + y_data**2) - np.sqrt(co_x_data**2 + co_y_data**2)
            zcoords_diff = z_data - co_z_data
        #print(len(radial))
        #try:
        #    naff_r_tune = pynaff_src.naff(radial,turns=47, nterms=1, skipTurns=0, getFullSpectrum=False)
        #    naff_r_tune = naff_r_tune[0][1]
        #    naff_z_tune = pynaff_src.naff(zcoords_diff,turns=47, nterms=1, skipTurns=0, getFullSpectrum=False)
        #    naff_z_tune = naff_z_tune[0][1]
        #except IndexError:
        #    naff_r_tune = np.NaN
        #    naff_z_tune = np.NaN

        if naff:
            #naff_r_tune = naff_shinji.naff(radial, 0, len(radial))
            #naff_z_tune = naff_shinji.naff(zcoords_diff, 0, len(zcoords_diff))
            #turns_ref = len(x_data)/steps_per_turn
            try:
                naff_r_tune = pnf.naff(radial - np.mean(radial), turns=60, nterms=1, skipTurns=0, getFullSpectrum=False)
                naff_r_tune = naff_r_tune[0][1]
                naff_z_tune = pnf.naff(zcoords_diff - np.mean(zcoords_diff), turns=60, nterms=1, skipTurns=0, getFullSpectrum=False)
                naff_z_tune = naff_z_tune[0][1]
                
                naff_r_tunes.append(naff_r_tune)
                naff_z_tunes.append(naff_z_tune)
            except (IndexError, ValueError) as e:
                pass
        else:
            w = np.abs(np.fft.rfft(radial))

            freqs = np.fft.rfftfreq(len(radial), tof)

            index_max_r = np.argmax(w)
            fft_r_tune_hz = freqs[index_max_r]

            z = np.abs(np.fft.fft(zcoords_diff))
            index_max_z = np.argmax(z)
            try:
                fft_z_tune_hz = freqs[index_max_z]
            except IndexError:
                fft_z_tune_hz = freqs[np.argmax(z[:len(z)//2])]
            fft_r_tune = fft_r_tune_hz*tof
            fft_z_tune = fft_z_tune_hz*tof

            fft_r_tunes.append(fft_r_tune)
            fft_z_tunes.append(fft_z_tune)

    
            #print(len(lost_particles))
            #print(naff_r_tune, "-----",naff_z_tune)
            # note that this is the fractional part of the tune

        #else:
            #print("id of lost particle = ",i)
            #lost_particles.append(i)
    comm.Barrier()

    naff_r_tunes = comm.gather(naff_r_tunes, root=0)
    naff_z_tunes = comm.gather(naff_z_tunes, root=0)
    lost_particles = comm.gather(lost_particles, root=0)

    fft_r_tunes = comm.gather(fft_r_tunes, root=0)
    fft_z_tunes = comm.gather(fft_z_tunes, root=0)


    #naff_r_tunes = [item for sublist in naff_r_tunes for item in sublist]
    #naff_z_tunes = [item for sublist in naff_z_tunes for item in sublist]
    #lost_particles = [item for sublist in lost_particles for item in sublist]
    if rank == 0:
        #print(naff_r_tunes)
        #print(naff_z_tunes)
        naff_r_tunes_gathered = []
        naff_z_tunes_gathered = []
        fft_r_tunes_gathered = []
        fft_z_tunes_gathered = []
        lost_particles_gathered = []

        for i, j, k, l, r in zip(naff_r_tunes, naff_z_tunes, lost_particles, fft_r_tunes, fft_z_tunes):
            naff_r_tunes_gathered = naff_r_tunes_gathered + i
            naff_z_tunes_gathered = naff_z_tunes_gathered + j
            fft_r_tunes_gathered = fft_r_tunes_gathered + l
            fft_z_tunes_gathered = fft_z_tunes_gathered + r
            lost_particles_gathered = lost_particles_gathered + k

        #naff_r_tunes = list(itertools.chain(*naff_r_tunes))
        #naff_z_tunes = list(itertools.chain(*naff_z_tunes))
        #lost_particles = list(itertools.chain(*lost_particles))
        print("num_lost_particles = ", len(lost_particles_gathered))
        print("tune calc time", MPI.Wtime()-tune_calc_time1)

    else:
        naff_r_tunes_gathered = None
        naff_z_tunes_gathered = None
        fft_r_tunes_gathered = None
        fft_z_tunes_gathered = None

    comm.Barrier()

    if naff:

        return naff_r_tunes_gathered, naff_z_tunes_gathered
    
    else:
        return fft_r_tunes_gathered, fft_z_tunes_gathered

def main(num_probes, probe_filepath, co_probe_filepath):
    if rank == 0:
        time1 = MPI.Wtime()

    probes = []
    co_probes = []

    for i in range(1, num_probes+1): 
        probe_df = read_dat.read_probe(filepath=probe_filepath, filename='PROBE'+str(i)+'.loss')
        #drop un used column to save memory
        probe_df.drop(['px ( )', 'py ( )', 'pz ( )', 'bunchNumber ( )'], axis=1, inplace=True)

        probes.append(probe_df)

        co_probes.append(read_dat.read_probe(filepath=co_probe_filepath, filename='PROBE'+str(i)+'.loss'))

    '''
    xcoords_allprobes =[]
    ycoords_allprobes =[]
    zcoords_allprobes =[]
    for dfprobes in probes:
        xcoords_allprobes.append(dfprobes['# x (m)'].to_numpy())
        ycoords_allprobes.append(dfprobes['y (m)'].to_numpy())
        zcoords_allprobes.append(dfprobes[' z (m)'].to_numpy())

    xcoords_allprobes =np.array(xcoords_allprobes, dtype=object)
    ycoords_allprobes =np.array(ycoords_allprobes, dtype=object)
    zcoords_allprobes =np.array(zcoords_allprobes, dtype=object)

    '''


    #CO_xcoords_allprobes =[]
    #CO_ycoords_allprobes =[]
    #CO_zcoords_allprobes =[]
    #for CO_dfprobes in co_probes:
    #    CO_xcoords_allprobes.append(CO_dfprobes['# x (m)'].to_numpy())
    #    CO_ycoords_allprobes.append(CO_dfprobes['y (m)'].to_numpy())
    #    CO_zcoords_allprobes.append(CO_dfprobes[' z (m)'].to_numpy())

    #CO_xcoords_allprobes =np.array(CO_xcoords_allprobes, dtype=object)
    #CO_ycoords_allprobes =np.array(CO_ycoords_allprobes, dtype=object)
    #CO_zcoords_allprobes =np.array(CO_zcoords_allprobes, dtype=object)


    # make dict of all the co coords at each probe so they can be looked up faster 
    #probe_df = probe_df[probe_df['id'] == 0]

    #radialco_dict  ={"probe"+str(1) : np.sqrt(CO_xcoords_allprobes[1][0]**2 + CO_ycoords_allprobes[1][0]**2)}
    #z_coord_co_dict  ={"probe"+str(1) : CO_zcoords_allprobes[1][0]}
    #for j in range(1, 16):
    #    radialco_dict["probe"+str(j+1)] = np.sqrt(CO_xcoords_allprobes[j][0]**2 + CO_ycoords_allprobes[j][0]**2)
    #    z_coord_co_dict["probe"+str(j+1)] = CO_zcoords_allprobes[j][0]
    
    #all_probes = [xcoords_allprobes, ycoords_allprobes, zcoords_allprobes]
    #tunex, tunez = probes_tune_calc(probes, co_probes, num_particles=int(1), naff=True)
    #print("naff tunex, naff tunez = ", tunex, tunez)

    naff = True
    tunex,  tunez = probes_tune_calc(probes, co_probes, naff=naff)
    if rank == 0:

        print("fft tunex, = ", tunex )
        print("fft tunez = ", tunez)
        fig, ax = plt.subplots()

        ax.scatter(np.array(tunex)*16, np.array(tunez)*16, marker='.')
        ax.scatter(3.41, 3.39, color='r', marker='.')


        if naff:
            np.save(probe_filepath+"naff_tunesx", tunex)
            np.save(probe_filepath+"naff_tunesz", tunez)
            ax.set_xlabel('naff tuneX')
            ax.set_ylabel('naff tuneZ')

        else:
            ax.set_xlabel('fft tuneX')
            ax.set_ylabel('fft tuneZ')

            np.save(probe_filepath+"fft_tunesx", tunex)
            np.save(probe_filepath+"fft_tunesz", tunez)


        print("total time", MPI.Wtime()-time1)

        np.save(probe_filepath+"tuneX.npy", tunex)
        print(probe_filepath)
        np.save(probe_filepath+"tuneZ.npy", tunez)

        plot_resonances = False
        if plot_resonances:
            resonance_plot(5, show_plot=False, fig_object=fig, axis_object=ax)
        else:
            plt.grid()

        #plt.xlim(left=3.3, right=3.5)
        #plt.ylim(bottom=3.3, top=3.65)
        #plt.savefig("tune_plot_kv_zoomin.png")
        plt.show()

if __name__=="__main__":

    main(16, filepaths["probe_filepath"], filepaths['bench_CO_dir'])


