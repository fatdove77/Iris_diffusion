"""

ORIGINAL AUTHOR: Libor Masek

CITATION: Libor Masek, Peter Kovesi. MATLAB Source Code for a Biometric Identification System Based on Iris Patterns. The School of Computer Science and Software Engineering, The University of Western Australia. 2003.

PYTHON IMPLEMENTATION BY: ROB HART OF SUSAN WALSH LAB.

"""

import numpy as np

class MasekGaborKernel(object):
    def __init__(self, ksize, wavelength, num_filters, multiplicitive_factor, sigmaOnf=0.5):
        self.__sigmaOnf = sigmaOnf

        self.wavelengths = self.__find_wavelengths(wavelength, num_filters, multiplicitive_factor)
        self.ksize = self.__find_ksize(ksize)
        self.radius = self.__find_radius(ksize)

        self.kernels, self.parameters = self.__create_kernels(self.__sigmaOnf, self.ksize, self.radius, self.wavelengths)


    def __find_ksize(self, ksize):
        new_size = ksize

        if new_size % 2 != 0:
            new_size = new_size - 1

        return new_size

    def __find_radius(self, ksize):
        radius = np.linspace(0, 0.5, int(ksize/2)+1)
        radius[0] = 1

        return radius
    
    def __find_wavelengths(self, wavelength, num_filters, multiplicitive_factor):
        wavelengths = []
        multiplicitive_factor = int(multiplicitive_factor)

        for i in range(num_filters):
            if i == 0:
                wavelengths.append(wavelength)
            else:
                wavelengths.append(wavelength * multiplicitive_factor * (i+1))

        return np.array(wavelengths, dtype=np.int32)
    
    def __create_kernels(self, signmaOnf, ksize, radius, wavelengths):
        kernels = []
        parameters = []

        for wavelength in wavelengths:
            frequency = 1 / wavelength
            
            kernel = np.exp((-(np.log(radius/frequency))**2) / (2 * np.log(signmaOnf)**2)) #create the log gabor kernel
            kernel[0] = 0
            kernel = np.pad(kernel, (0, int(ksize/2)-1), 'constant') #pad the kernel with 0s to make it the same length as the number of columns
        
            kernels.append(kernel)
            parameters.append((ksize, signmaOnf/wavelength, 'N/A', wavelength, 'N/A', 'N/A'))

        return np.array(kernels), np.array(parameters)
    

class MasekGaborResponse(object):
    def __init__(self, image, kernels):
        self.__image = self.__adjust_img_size(image)
        self.responses = self.__apply_filters(self.__image, kernels)
        self.quantized_responses = self.__quantize_responses(self.responses)
        self.iris_barcode = (self.quantized_responses*255).astype(np.uint8)

    def __adjust_img_size(self, image):
        if image.shape[1] % 2 != 0:
            image = image[:, :-1]

        return image

    def __apply_filters(self, image, kernels):
        responses = []

        for kernel in kernels:
            results = np.zeros((image.shape[0], image.shape[1]), dtype=np.complex64) #index 0 is the real part, index 1 is the imaginary part

            for row_num in range(image.shape[0]):
                features = image[row_num, :]
                feature_signal = np.fft.fft(features)
                feature_vector = np.fft.ifft(feature_signal * kernel)
                results[row_num, :] = feature_vector

            responses.append(results)

        return np.array(responses)
    
    def __quantize_responses(self, responses):
        quantized_responses = []

        replacements = np.array([[1,1],[0,1],[0,0],[1,0]], dtype='?')
        for response in responses:
            quantized_response = np.array([np.empty(response.shape), np.empty(response.shape)], dtype='?')
            response_phase = np.angle(response)
            conditions = [(response_phase >= 0) & (response_phase < np.pi/2), (response_phase >= np.pi/2) & (response_phase < np.pi), (response_phase >= -np.pi) & (response_phase < -np.pi/2), (response_phase >= -np.pi/2) & (response_phase < 0)]
            for i, condition in enumerate(conditions):
                quant_indices = np.where(condition)
                quantized_response[0][quant_indices] = replacements[i][0]
                quantized_response[1][quant_indices] = replacements[i][1]
            
            quantized_responses.append(quantized_response)
            
        return np.array(quantized_responses, dtype='?')