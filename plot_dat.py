from read_dat import read_FieldMapRPHI, read_trackOrbit, read_Angle, read_FieldMap, read_probe, read_FieldMapRPHI
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


import json

with open("configs.json", "r") as jsonfile:
    configs = json.load(jsonfile)

filepaths = configs['filepaths']
filenames = configs['filenames']



root_path = filepaths['root_path']

#root_path = "/home/carl/Documents/hFFA/example_1/example_sector_ffa/"
track_filename = "DF_lattice" + filenames['track-orbit']
#track_filename = "FFA_FODO_lattice" + filenames['track-orbit']
#track_filename = filenames['DF_hFFA_inputfile_bench'] + filenames['track-orbit']

fieldmap_filename = "data/"+filenames['XYfieldmap']
fieldmap_cyclindrical_filename =  "data/"+filenames['RPHIfieldmap']

trackOrbit_df = read_trackOrbit(filepath=root_path
                                , filename=track_filename)

#Angle_df = read_Angle(filepath=root_path
#                    , filename=angle_filename)

FieldMap_df = read_FieldMap(root_path, filename=fieldmap_filename)
FieldMapRPHI_df = read_FieldMapRPHI(root_path, filename=fieldmap_cyclindrical_filename)

kgauss2Tesla = 1e3/1e4
    

def field_hist(FieldMap_df):
    min_field = FieldMap_df['Bz [kGauss]'].min()*1e3/1e4
    max_field = FieldMap_df['Bz [kGauss]'].max()*1e3/1e4
    if min_field < 0:
        abs_max_field = max(max_field, abs(min_field))
        field_norm = matplotlib.colors.Normalize(-abs_max_field, abs_max_field, False)
        cmap_ = "bwr"
    else:
        field_norm = matplotlib.colors.Normalize(0, max_field, False)
        cmap_ = "Reds"

    return plt.hist2d(FieldMap_df['x [m]'], 
                FieldMap_df['y [m]'], 
                weights=FieldMap_df['Bz [kGauss]']*1e3/1e4,
                bins=[int(len(FieldMap_df['x [m]'])/1500), int(len(FieldMap_df['y [m]'])/1500)],
                cmap=cmap_,
                norm=field_norm)


def plot_field_map_and_orbit(FieldMap_df, trackOrbit_df):

    field_hist(FieldMap_df=FieldMap_df)
    plt.plot(trackOrbit_df['x [m]'], trackOrbit_df['y [m]'],linewidth=1, color='g')
    print('mean radius = ',np.mean(np.sqrt(trackOrbit_df['x [m]']**2 + trackOrbit_df['y [m]']**2)))
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.colorbar()
    plt.show()
    #plt.savefig('x-y-Bz-hist.png')
    #plt.show()
    plt.close()

def plot_field_map_and_orbit_withprobe(FieldMap_df, probes):

    xcoords =[]
    Pxcoords =[]
    ycoords =[]
    Pycoords =[]
    for df in probes:
        xcoords.append(df['# x (m)'].to_numpy()[0])
        ycoords.append(df['y (m)'].to_numpy()[0])

    field_hist(FieldMap_df=FieldMap_df)
    plt.plot(trackOrbit_df['x [m]'], trackOrbit_df['y [m]'],linewidth=1, color='g')
    plt.scatter(xcoords, ycoords)
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.colorbar()
    #plt.axes().set_aspect('equal')
    #plt.savefig('x-y-Bz-hist.png')
    plt.show()
    plt.close()


def plot_phase_space(E_kin, x_co=3.9910649675642533, px_co=-0.010031648542825716):
    plt.figure(1)
    plt.subplot(211)
    m_0 = 938.272 
    gamma = (E_kin/m_0) +1
    beta = (1- (1/gamma**2))**0.5
    df = read_probe(filepath=filepaths['root_path'], filename='PROBE'+str(1)+'.loss')

    plt.scatter((df['# x (m)'].to_numpy())*1000, (df['px ( )'].to_numpy())*1000/(beta*gamma), marker='.')
    plt.xlabel("x [mm]")
    plt.ylabel("Px [mrad]")

    plt.subplot(212)
    plt.scatter(df[' z (m)'].to_numpy()*1000, (df['pz ( )'].to_numpy())*1000/(beta*gamma), marker='.')
    plt.xlabel("z [mm]")
    plt.ylabel("Pz [mrad]")
    plt.show()

    GX = df['py ( )'].to_numpy()/beta # units of gamma
    #GY = df['py ( )'].to_numpy()/beta # units of gamma
    #GZ = df['pz ( )'].to_numpy()/beta # units of gamma

    KE_X = (GX - 1)*m_0
    #KE_Y = (GY - 1)*m_0
    #KE_Z = (GZ - 1)*m_0
    energy = np.sqrt(KE_X**2)

    time=df['time (s)'].to_numpy()
    plt.plot(time, energy)
    plt.show()

    print("x_co, px_co = ", x_co, px_co)

    xcoords = df['# x (m)'].to_numpy() - x_co
    pxcoords = df['px ( )'].to_numpy()/(beta*gamma) - px_co/(beta*gamma)
    zcoords = df[' z (m)'].to_numpy()
    pzcoords = df['pz ( )'].to_numpy()/(beta*gamma)


    varianceX = np.mean(xcoords**2)
    varianceZ = np.mean(zcoords**2)

    variancePx = np.mean(pxcoords**2)
    variancePz = np.mean(pzcoords**2)

    varianceXPx = np.mean(xcoords*pxcoords)
    varianceZPz = np.mean(zcoords*pzcoords)

    rms_emitX = np.sqrt(varianceX*variancePx - varianceXPx**2)
    rms_emitZ = np.sqrt(varianceZ*variancePz - varianceZPz**2)

    # calc twiss params
    betaX = varianceX/rms_emitX
    betaZ = varianceZ/rms_emitZ

    gammaX = variancePx/rms_emitX
    gammaZ = variancePz/rms_emitZ

    alphaX = -varianceXPx/rms_emitX
    alphaZ = -varianceZPz/rms_emitZ

    print(betaX*gammaX - alphaX**2)
    print(betaZ*gammaZ - alphaZ**2)

    print('emitX = ',rms_emitX)
    print('emitZ = ',rms_emitZ)

    print('betaX = ', betaX)
    print('betaZ = ', betaZ)
    print('gammaX = ', gammaX)
    print('gammaZ = ', gammaZ)
    print('alphaX = ',alphaX)
    print('alphaZ = ',alphaZ)

def main():

    plot_field_map_and_orbit(FieldMap_df, trackOrbit_df)

    plot_phase_space(3)
    probes = []
    num_cells = 16# number of probes must be the same as the number of cells
    for i in range(1,num_cells+1): 
        df = read_probe(filepath=filepaths['root_path'], filename='PROBE'+str(i)+'.loss')
        probes.append(df)

    plot_field_map_and_orbit_withprobe(FieldMap_df=FieldMap_df, probes=probes)

if __name__ == "__main__":
    main()
