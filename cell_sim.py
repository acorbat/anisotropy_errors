import numpy as np
from scipy import stats
import imreg_dft as dft

import anisotropy_functions as af


class Cell():
    
    def __init__(self, proteins, poisson, size=500):
        self.proteins = proteins
        self.poisson = poisson
        self.cell_image, self.mask = self.create_cell(size)
        
        
    def create_cell(self, size):
        image = np.zeros((size, size))
        half_point = np.floor(size // 2).astype(int)
        initial_point = np.floor(0.2 * size).astype(int)

        gradient = np.arange(0, half_point-initial_point + 1) / (half_point-initial_point + 1)

        image[initial_point:half_point + 1, 
              initial_point:size-initial_point] = np.tile(gradient,
                                                          (size-2*initial_point, 1)).T
        image[size-initial_point:half_point - 1:-1, 
              initial_point:size-initial_point] = np.tile(gradient, 
                                                          (size-2*initial_point, 1)).T
        image = image * image.T

        if self.poisson:
            image = stats.poisson.rvs(image * self.proteins)
        else:
            image *= self.proteins
            
        mask = np.zeros_like(image)
        mask[np.nonzero(image)] = 1
        
        return np.floor(image).astype(int), mask.astype(bool)
    
    
    def add_biosensor(self, biosensor):
        if isinstance(biosensor, dict):
            self.biosensor = Biosensor(**biosensor)
        elif isinstance(biosensor, Biosensor):
            self.biosensor = biosensor
        else:
            raise TypeError('Dictionary or Biosensor expected')
            
    
    def generate_fraction_image(self, anisotropy):
        self.anisotropy = anisotropy
        
        monomer_fraction = self.biosensor.get_monomer_fraction(anisotropy)
        fraction_image = stats.binom.rvs(self.cell_image, monomer_fraction)
        fraction_image = fraction_image.astype(float)
        nonzeros = self.mask
        
        fraction_image[nonzeros] = fraction_image[nonzeros] / self.cell_image[nonzeros]
        
        self.fraction_image = fraction_image
    
    
    def generate_intensity_images(self, anisotropy=None):
        if anisotropy is not None:
            self.generate_fraction_image(anisotropy)
        
        nonzeros = self.mask
        self.parallel_image = np.zeros_like(self.fraction_image)
        self.parallel_image[nonzeros] = af.intensity_parallel_from_monomer(self.fraction_image[nonzeros], 
                                                                           self.biosensor.anisotropy_monomer, 
                                                                           self.biosensor.anisotropy_dimer, 
                                                                           self.biosensor.b, 
                                                                           self.cell_image[nonzeros] * self.biosensor.brightness)
        
        self.perpendicular_image = np.zeros_like(self.fraction_image)
        self.perpendicular_image[nonzeros] = af.intensity_perpendicular_from_monomer(
            self.fraction_image[nonzeros],
            self.biosensor.anisotropy_monomer, 
            self.biosensor.anisotropy_dimer, 
            self.biosensor.b, 
            self.cell_image[nonzeros] * self.biosensor.brightness)

            


class Biosensor():
    
    def __init__(self, anisotropy_monomer, anisotropy_dimer, delta_b):
        self.anisotropy_monomer = anisotropy_monomer
        self.anisotropy_dimer = anisotropy_dimer
        self.delta_b = delta_b
        self.b = 1 + self.delta_b
        self.brightness = 10
        
    def get_monomer_fraction(self, anisotropy):
        return af.monomer_from_anisotropy(anisotropy, 
                                          self.anisotropy_monomer, self.anisotropy_dimer, 
                                          self.b)


class Microscope():
    
    def __init__(self):
        self.shift = (4, 6)
        self.bkg_distrib = stats.norm
        self.acq_distrib = stats.norm
    
    
    def add_background(self, image, bkg=200, std=10):
        if std != 0:
            bkg = self.bkg_distrib.rvs(np.zeros_like(image) + bkg, std)
        
        return image + bkg
    
    
    def add_acquisition_noise(self, image, std=10):
        noise = self.acq_distrib.rvs(np.zeros_like(image), std)
        return image + noise
    
    
    def add_shift(self, image):
        return dft.transform_img(image, tvec=self.shift, mode='nearest')
    
    
    def acquire_images(self, parallel, perpendicular):
        parallel = parallel.copy()
        parallel = self.add_background(parallel)
        parallel = self.add_acquisition_noise(parallel)
        
        perpendicular = perpendicular.copy()
        perpendicular = self.add_background(perpendicular)
        perpendicular = self.add_acquisition_noise(perpendicular)
        
        perpendicular = self.add_shift(perpendicular)
        
        return parallel, perpendicular
    
    
    def acquire_cell(self, cell):
        parallel = cell.parallel_image.copy()
        perpendicular = cell.perpendicular_image.copy()
        
        return self.acquire_images(parallel, perpendicular)
    

class Corrector():
    
    def __init__(self):
        self.bkg_correction = subtract_constant
        self.bkg_params = {'bkg_value': 200}
        self.shift = (-4, -6)
        
        
    def correct(self, parallel, perpendicular):
        parallel = parallel.copy()
        perpendicular = perpendicular.copy()
        
        parallel = self.bkg_correction(parallel, **self.bkg_params)
        perpendicular = self.bkg_correction(perpendicular, **self.bkg_params)
            
        perpendicular = dft.transform_img(perpendicular, tvec=self.shift, mode='nearest')
        
        return parallel, perpendicular
        
        
def subtract_constant(image, bkg_value):
    return np.clip(image - bkg_value, 0, np.inf)