from pandas import DataFrame
from scipy.optimize import minimize
import os 
import read_dat
import numpy as np
import time
import json
from opal_input_file import OpalInputFile
with open("configs.json", "r") as jsonfile:
    configs = json.load(jsonfile)
    filepaths = configs['filepaths']
    filenames = configs['filenames']



#def write_dist_file(filename, x, y, z, Px='0.0',Py='0.0',Pz='0.0'):#
#
#    with open(filepath +  filename, 'w') as f:
#        f.write('1') # 1 particle
#        f.write(str(x)+ ' '+str(Px)+' '+ str(y)+ ' '+str(Py)+' '+ str(z)+' '+ str(Pz))
#        f.close()

class ClosedOrbitFinder():
    def __init__(self, *args):
        self.k_value = args[0]
        self.F_mag_field = args[1]
        self.D_mag_field = args[2]
        self.energy = args[3]
        self.probes = None

    def run_opal(self, xPxcoords,steps_per_turn=2850, filepath=filepaths["root_path"], filename=filenames['FD_16_cell_benchmark'], z=0.0,Pz=0.0):
        '''
        run opal input file above and then find CO by optimizing the parameters in xPxcoords, more paramters 
        may be added. 
        Cell opening angle is in deg
        function minimizes the difference in x Px and y Py at the start and end of each cell in on turn
        ie minimizes the change in radius and radiual momentum from the start of a cell to the end.

        ''' 
        dist_filename = filenames["12MeV_CO_coords"]


        num_turns = 1
        num_particles = 1
        create_trackorbit = 0
        Field_map = False
        include_RF = False

        OpalInputFile(num_turns, dist_filename, create_trackorbit, steps_per_turn, num_particles, Field_map, include_RF, self.energy, self.k_value, self.F_mag_field, self.D_mag_field).benchamrk_lattice(filepath, filename)

        with open(filepath +  dist_filename, 'w') as f:

            f.write('1\n') # 1 particle
            f.write(str(xPxcoords[0])+ ' '+str(xPxcoords[1])+' '+ str(0.0)+ ' '+str(0.0)+' '+ str(z)+' '+ str(Pz))
            f.close()
        print('input = ', xPxcoords)

        os.system(filepaths["opal_exe_path"] +' --info 0 ' + filepath + filename)


        # read probes
        self.probes = []
        num_cells = 16# number of probes must be the same as the number of cells
        for i in range(1,num_cells+1):
            try:
                self.probes.append(read_dat.read_probe(filepath=filepaths["probe_filepath"], filename='PROBE'+str(i)+'.loss'))
                os.remove(filepaths["probe_filepath"]+'PROBE'+str(i)+'.loss')
            except FileNotFoundError:
                self.probes.append(DataFrame(data={'# x (m)':[np.random.rand()*10000], 'y (m)':[np.random.rand()*10000], 'px ( )':[np.random.rand()*10000], 'py ( )':[np.random.rand()*10000]}))

        xcoords =[]
        Pxcoords =[]
        ycoords =[]
        Pycoords =[]
        for df in self.probes:
            xcoords.append(df['# x (m)'].to_numpy()[0])
            ycoords.append(df['y (m)'].to_numpy()[0])
            Pxcoords.append(df['px ( )'].to_numpy()[0])
            Pycoords.append(df['py ( )'].to_numpy()[0])
        xcoords =np.array(xcoords)
        ycoords =np.array(ycoords)
        Pycoords =np.array(Pycoords)
        Pxcoords =np.array(Pxcoords)

        Pxcoords_rot =[]
        Pycoords_rot =[]
        phi = np.arctan2(ycoords, xcoords) +np.pi

        for k,n in enumerate(phi):
            Pxcoords_rot.append(np.cos(n)*Pxcoords[k] - np.sin(n)*Pycoords[k])
            # Px_arr is chnaging here so need a new variable
            Pycoords_rot.append(np.sin(n)*Pxcoords[k] + np.cos(n)*Pycoords[k])

        Pxcoords_rot =np.array(Pxcoords_rot) # do I need to rotate the momentum coords?
        Pycoords_rot =np.array(Pycoords_rot)


        #import matplotlib.pyplot as plt
        #plt.scatter(xcoords, ycoords)
        #plt.axes().set_aspect('equal')
        #plt.show()
        print('len xcoords = ',len(xcoords))
        print(xcoords)
        rcoords = xcoords*xcoords + ycoords*ycoords
        diff=0
        for i in range(len(rcoords)):
            diffr = (rcoords[i] - rcoords[0])**2
            diffPx = (Pxcoords_rot[i] - Pxcoords_rot[0])**2
            diffPy = (Pycoords_rot[i] - Pycoords_rot[0])**2
        
            diff += np.sqrt(diffr + diffPx + diffPy)


        #diffr = abs(np.sum(np.diff(rcoords)))
        #diffPr = abs(np.sum(np.diff(Pxcoords_rot)) + np.sum(np.diff(Pycoords_rot) ))
        #diffPr = abs(np.sum(np.diff(Pxcoords*Pxcoords + Pycoords*Pycoords)))
        #iffPr = abs(np.sum(np.diff(Pxcoords))+ np.sum(np.diff(Pycoords)))
        #diff = diffr +diffPr
        print('diffr = ', diffr)
        #print('diffPr = ', diffPr)
        print('diff =', diff)
        return diff

    def main(self, initial_x, initial_px):
        t1 = time.time()
        #minimize(run_opal, x0=[5.051859350125676, -0.009517835658783173], method='Nelder-Mead', options={'disp':True})
        #minimize(run_opal, x0=[4.0, 0.0], method='Powell', bounds=((3.5,4.5),(-0.05,0.05)), options={'disp':True, 'xtol':1e-7, 'ftol':1e-7})
        #minimize(run_opal, x0=[4.0, 0.0], method='L-BFGS-B', bounds=((4.0,4.5),(-0.01,0.01)), options={'disp':True})
        res = minimize(self.run_opal, x0=[initial_x, initial_px], method='Nelder-Mead', options={'disp':True, 'xatol':1e-6, 'fatol':1e-6})

        t2 = time.time()
        print('time taken = ',t2-t1)
        
        return self.probes, res


if __name__ == "__main__":
    ClosedOrbitFinder(6.6841968722339296, -0.2778341842733474, 0.10090213357455474).main(4.0, 0.0)

