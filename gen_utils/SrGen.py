import os
from tqdm import tqdm
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import cv2
import math as math
import nibabel as nib
import pydicom
from skimage.transform import rotate, AffineTransform, warp, rescale, resize

class SrGen:
    def __init__(self, inp_dir, HR_out_dir, LR_out_dir, prefix='', suffix=''):
        self.inp_dir = inp_dir
        self.HR_out_dir = HR_out_dir
        self.HR_files = None
        self.LR_out_dir = LR_out_dir
        self.LR_files = None
        self.inp_files, self.inp_paths = self._get_inp_(prefix, suffix)
        self.template = self.get_template()

### ORANIZATIONAL METHODS
    def _get_inp_(self, prefix='', suffix=''):

        files = []
        paths = []
        # If they have provided a list of directories (in the case of DICOM or scattered data)
        if isinstance(self.inp_dir, list):
            for inp_dir in self.inp_dir:
                for fil in os.listdir(inp_dir):
                    if fil.startswith(prefix) and fil.endswith(suffix):
                        paths.append(inp_dir + fil)
                        files.append(fil)

                    if not files:
                        raise FileNotFoundError('No applicable files found in input directory')
        else:
            for fil in os.listdir(self.inp_dir):
                if fil.startswith(prefix) and fil.endswith(suffix):
                    paths.append(self.inp_dir + fil)
                    files.append(fil)

                if not files:
                    raise FileNotFoundError('No applicable files found in input directory')

        return files, paths
    
    def _get_LR_out_(self):
        # get list of files in output directory and determine matching files
        return os.listdir(self.LR_out_dir)

    def _get_HR_out_(self):
        return os.listdir(self.HR_out_dir)

    def get_template(self):
        try:
            return self.template
        except:
            return {'out_type':'png',
                'unit':'intensity', #Currently only matters for png
                'resolution':None,
                'same_size': True, # Whether to have the LR image be the same size as the HR image
                                   # (i.e. whether to scale down then up or just down)
                'translation':None, # Have both single value or multiple
                'rotation': None, # Around each axis
                'scale': False, # What magnitude to zoom in for added jitter
                'patch': False, # Have this accept 3 dimensional input [x,y,z], [x,y], or single
                'step': 10, # Also have this accept 3 dimensional input
                'keep_blank': False,
                'blank_ratio': 0.4,
            }
    
    def set_template(self, temp):
        self.template = temp


    def match_altered(self, update=True, paths=False, sort=False):
        # Get the files that have been generated in the output directory
        # If update is false, then just return a list of matched names, if true then
        # change the class variable values accordingly.
        hr_files = self._get_HR_out_()
        lr_files = self._get_LR_out_()

        # Get a set of all the files with agreement before the metadata
        if len(hr_files) > len(lr_files):
            if sort: #TODO: make sort so it isnt [*1.*, *10.*, *100.*, ..., *2.*,...]
                matches = sorted(list(set(hr_files)-(set(hr_files)-set(lr_files))))
            else:
                matches = list(set(hr_files)-(set(hr_files)-set(lr_files)))
        else:
            if sort:
                matches = sorted(list(set(lr_files)-(set(lr_files)-set(hr_files))))
            else:
                matches = list(set(lr_files)-(set(lr_files)-set(hr_files)))

        if update:
            # If you want to save these matched files as class variables
            self.HR_files = [self.HR_out_dir + _ for _ in matches]
            self.LR_files = [self.LR_out_dir + _ for _ in matches]
            print('HR and LR file locations updated')
        
        if paths:
            return self.HR_files, self.LR_files

    def change_out(self, HR_out_dir, LR_out_dir):
        # Change the output locations so you can save into a new file
        self.HR_out_dir = HR_out_dir
        self.HR_files = None
        self.LR_out_dir = LR_out_dir
        self.LR_files = None

### ACTUAL IMAGE MANIPULATION BELOW

    def run(self, clear=False, save=False):
        # This method is called to generate the data

        if clear:
            print('Clearing existing output directories')
            shutil.rmtree(self.HR_out_dir, ignore_errors=True)
            if self.template['resolution'] != None:
                shutil.rmtree(self.LR_out_dir, ignore_errors=True)
                os.makedirs(self.LR_out_dir, exist_ok=True)

        os.makedirs(self.HR_out_dir, exist_ok=True)
        fnames_h = []
        fnames_l = []

        for ids, im in enumerate(self.inp_paths):
            im_h = self.load_image(im)
            opp, im_h = self.img_transform(im_h)

            # Prevents weird new file naming issues when input is compressed file (.nii.gz)
            im = Path(im)
            while im.suffix in {'.tar', '.gz', '.zip'}:
                im = im.with_suffix('')
            
            im = os.path.splitext(im)[0]+opp # Add transformations to file name
            im = os.path.split(im)[1]
        
            # Generate Low Resolution
            if self.template['resolution']:

                dim = im_h.shape
                # efficient way to either make a single value into an array or do nothing if resolution is already a vector
                # TODO: replace transformation if statements with this
                self.template['resolution'] = [int(x) for x in np.multiply(np.ones(len(dim)), self.template['resolution'])]

                # Check that dimensions of HR image are multiples of resolution change, else shave off data
                print(dim)
                print(self.template['resolution'])
                for i in range(len(dim)):
                    if dim[i] % self.template['resolution'][i]:
                        # If it isn't a clean scaling down
                        _ = dim[i]-(dim[i] % self.template['resolution'][i])

                        im_h = np.delete(im_h,[x for x in range(_, dim[i])],i)

                im_l = self.gen_LR_img(im_h, self.template['resolution'])

            # Create image patches and save them
            if self.template['patch'] and save:
                fnames_h, slice_select = self.img2patches(im_h, self.HR_out_dir + im, save=True, sanity_check=True)
                if self.template['resolution'] != None:
                    fnames_l = self.img2patches(im_l, self.LR_out_dir + im, same_size=self.template['same_size'], save=True, slice_select=slice_select, sanity_check=False)

                    # if not _a == _:
                    #     raise FileExistsError('''WARNING: The patches for High and Low resolution do not match, this is
                    #         most likely due to resolution scaling or patches/steps not being divisible by resolution''')

            elif save:
                fname_h = self.HR_out_dir + im
                self.save_image(fname_h, im_h)
                fnames_h.append(fname_h)
                if self.template['resolution']:# != None:
                    fname_l = self.LR_out_dir + im
                    self.save_image(fname_l, im_l)
                    fnames_l.append(fname_l)


        self.HR_files = fnames_h
        self.LR_files = fnames_l

        print('Files processed successfully')



    def load_image(self, im_path, verbose=False):
        # Given an image path, determines the function required to load the contents
        # as a numpy array, which is returned.
        fil_typ = os.path.splitext(im_path)[1]

        if fil_typ == '.png':
            # If file is a png
            img = np.array(Image.open(im_path))
            if verbose:
                print(f'Loading {im_path} as png')
                print(f'Image shape:{img.shape}')

            if self.template['unit'] == 'intensity':
                img = self.rgb2ycrbcr(img)
                img = img[:,:,0] #Just deal with intensity values at the moment because 
                                 # having multiple channels throws off cv2 when saving, 
                                 # since it also does BGR instead of RGB and will save a blue image
            elif self.template['unit'] == 'color':
                pass

        elif fil_typ == '.nii' or fil_typ == '.gz':
            img = nib.load(im_path).get_fdata()
            if verbose:
                print(f'Loading {im_path} as nii')
                print(f'Image shape:{img.shape}')

        elif fil_typ == '.dcm':
            img = pydicom.dcmread(im_path).pixel_array
            if verbose:
                print(f'Loading {im_path} as dicom')
                print(f'Image shape:{img.shape}')

        else:
            raise FileNotFoundError(f'Image file type {fil_typ} not supported.')

        return img

    def gen_LR_img(self, im_h, res, interp=1):
        # Generate the low-resolution image from the corresponding HR image using resizing
        dim = im_h.shape

        # TODO: Patching error occurs when the shape of an image has dimension of odd magintude with
        #       even 'res', and vice versa. Need to come up with a fix for this...
        new_dims = [math.floor(x) for x in np.divide(dim, res)]

        im_l = resize(im_h, new_dims, order = interp, mode='symmetric')

        if self.template['same_size']:
            im_l = resize(im_l, dim, order= interp, mode = 'symmetric')

        return im_l

    def img_transform(self, im_h, save=False):
        # Transform the original files using a variety of methods
        # Have im_h and im_l be numpy arrays
        opp = '' #string for storing the operations performed on the images

        dim = im_h.shape
        if len(dim)>3:
            raise ValueError('Dimension of input data not currently supported')
        
        # If single image is provided for any of these settings, convert into list of N dimensions
        if self.template['translation'] == None:
            trans = [None]
        elif type(self.template['translation']) != list:
            trans = [self.template['translation'] for _ in range(len(dim))]
        else:
            trans = self.template['translation'][:] #Weird thing I have to add to not link changes to 'trans' to self.template
        
        for idx, x in enumerate(trans):
            if trans[idx] != 0 and trans[idx] != None:
                trans[idx] = np.random.randint(-x,x)


        # Rotation
        if self.template['rotation'] == None:
            rot = [None]
        elif type(self.template['rotation']) != list:
            rot = [self.template['rotation'] for _ in range(len(dim))]
        else:
            rot = self.template['rotation'][:]

       
        for idx, x in enumerate(rot):
            if x != 0 and x != None:
                rot[idx] = np.random.randint(-x,x)


        # Scaling
        if self.template['scale'] == None:
            scale = [None]
        elif type(self.template['scale']) != list:
            scale = [self.template['scale'] for _ in range(len(dim))]
        else:
            scale = self.template['scale'][:]

        for idx, x in enumerate(scale):
            if scale[idx] != None and scale[idx] > 1:
                scale[idx] = np.random.randint(1,x+1)
            else:
                scale[idx] = 1



        # TODO: Issue with low resolution not necessairly having the same dimensions
        # Rotation 2D
        if len(rot) == 2 and type(rot[0])==int: #Make sure rotation isn't None for some reason
            im_h = rotate(im_h, rot[0], order=1)
            opp += f'_rot{rot[0]}'
        # Rotation 3D
        elif len(rot) == 3:
            for i in range(dim[0]):
                im_h[i,:,:] = rotate(im_h[i,:,:],rot[0], order=1)
            for i in range(dim[1]):
                im_h[:,i,:] = rotate(im_h[:,i,:],rot[1], order=1)
            for i in range(dim[2]):
                im_h[:,:,i] = rotate(im_h[:,:,i],rot[2], order=1)
            opp += f'_rot{rot[0]}_{rot[1]}_{rot[2]}'
            

        # Translation
        if len(trans) == 2:
            transform = AffineTransform(translation=(trans[0], trans[1]))
            im_h = warp(im_h, transform, mode="symmetric")
            opp += f'_tr{trans[0]}_{trans[1]}'
        elif len(trans) == 3:
            transform = AffineTransform(translation=(trans[1], trans[2]))
            for i in range(dim[0]):
                im_h[i,:,:] = warp(im_h[i,:,:],transform, mode = 'symmetric')

            for i in range(dim[1]):
                # Because two dimensions were already translated, you only need to translate
                # along one dimension
                im_h[:,i,:] = warp(im_h[:,i,:], AffineTransform(translation=(trans[0],0)), mode='symmetric')
                
            opp += f'_tr{trans[0]}_{trans[1]}_{trans[2]}'

        # Scaling
        if len(scale) >= 2:
            #transform = AffineTransform(scale=_a)
            im_h = rescale(im_h, scale = scale, mode='symmetric')
            
            # Dumb way to make added string fit based on scaling
            try: opp+= f'_scale_{scale[0]}_{scale[1]}_{scale[2]}'
            except: opp+= f'_scale_{scale[0]}_{scale[1]}'

        return opp, im_h

    def img2patches(self, im_h, fname, same_size=True, keep_blank=False, slice_select=None, save=False, sanity_check=False):
        # Depending on the number of dimenions in the `patch` value, either make 2D
        # or 3D images

        dim = im_h.shape
        print(f'shape of image = {dim}')
        patch_size = self.template['patch'][:]
        step = self.template['step'][:]

        
        im_name=Path(fname).with_suffix('').__str__()
        #im_name = fname.split('.')[:-2][0] #Kind of janky way to just strip away the suffix
        
        if slice_select: #If slice_select is provided, then getting rid of blanks really screws things up
            print('keeping blank')
            keep_blank = True

        if type(patch_size) != list:
            patch_size = [patch_size for _ in range(len(dim))]
        
        if type(step) != list:
            step = [step for _ in range(len(dim))]

        # Whether to shrink the patch size and step size down by the scaling amount for LR images without the same dimensions as the HR images
        if not same_size:
            try: 
                patch_size = [math.floor(x) for x in np.divide(patch_size,self.template['resolution'])]
                print(f'patch size = {patch_size}')
            except: raise ValueError(f'Resolution change coefficient: {self.template["resolution"]} not defined properly for patch_size: {patch_size}')

            try: 
                step = [math.floor(x) for x in np.divide(step,self.template['resolution'])]
                print(f'step size = {step}')
            except: raise ValueError(f'Resolution change coefficient: {self.template["resolution"]} not defined properly for step: {step}')
        else:
            print(f'patch size = {patch_size}')
            print(f'step size = {step}')

        # Create a numpy stack following Pytorch protocols, so 1 dimension more than patch
        
        # Count number of non-zero entries
        cnt = 0
        blank = 0
        not_blank = []
        itter = -1

        # Get total number of patches that will be created:
        #patch_count = np.prod([len(range(0,i,step[idx])) for idx, i in enumerate(dim)])
        print(f'patch guess = {np.prod([math.floor((i-patch_size[idx])/step[idx])+1 for idx,i in enumerate(dim)])}')
        patch_count = np.prod([math.floor((i-patch_size[idx])/step[idx])+1 for idx,i in enumerate(dim)])
        patch_vol = math.prod(patch_size)*self.template['blank_ratio']

        if len(dim) == 2:
            stack = np.zeros(patch_count,patch_size[0],patch_size[1])
            print(f'stack size = {stack.shape}')

            for i in range(0,dim[0],step[0]):
                for j in range(0,dim[1],step[1]):
                    if i+patch_size[0] <= dim[0] and j+patch_size[1] <= dim[1]:
                        itter = itter+1 #just a calculator for finding when blanks occur
                        samp = im_h[i:i+patch_size[0],j:j+patch_size[1]]

                        if keep_blank or (samp==0).sum() <= patch_vol:#(samp.max() > 0):
                            stack[cnt,:,:] = samp
                            cnt += 1
                            not_blank.append(itter)
                        else:
                            blank += 1
                            #blank.append(_)
        elif len(dim) == 3:
            stack = np.zeros((patch_count,patch_size[0],patch_size[1], patch_size[2]))
            print(f'stack size = {stack.shape}')

            for i in range(0,dim[0],step[0]):
                for j in range(0,dim[1],step[1]):
                    for k in range(0,dim[2],step[2]):
                        #itter = itter+1 #just a calculator for finding when blanks occur
                        if i+patch_size[0] <= dim[0] and j+patch_size[1] <= dim[1] and k+patch_size[2] <= dim[2]:
                            itter = itter+1
                            samp = im_h[i:i+patch_size[0],j:j+patch_size[1], k:k+patch_size[2]]

                            if keep_blank or (samp==0).sum() <= patch_vol:#(samp.max() > 0):
                                stack[cnt,:,:,:] = samp
                                cnt += 1
                                not_blank.append(itter)
                            else:
                                blank += 1
                                #blank.append(_)
            print(itter)
        else:
            raise IndexError(f'Images of dimension {dim} not supported by this method. Only 2D and 3D data accepted.')
        

        #TODO: There MUST be a better way to organize this whole mess, lol

        fnames = []
        if slice_select:
            for i in range(len(slice_select)):
                fnames.append(f'{im_name}_{i}.{self.template["out_type"]}')
        else:
            for i in range(cnt):
                fnames.append(f'{im_name}_{i}.{self.template["out_type"]}')

        if save:
            if slice_select:
                for idx, i in tqdm(enumerate(slice_select)):
                    self.save_image(fnames[idx], stack[i], verbose=False)
            else:
                for idx, i in tqdm(enumerate(fnames)):
                    self.save_image(i,stack[idx], verbose=False)
            if sanity_check:
                print(f'Number of patches: {len(not_blank)}')
                print(f'Number of blank patches: {blank}')
                return fnames, not_blank
            else:
                return fnames
        else:
            if sanity_check:
                return fnames, stack, not_blank
            else:
                return fnames, stack

    def rgb2ycrbcr(self, img_rgb):
        #Takes an RBG image and returns it as a YCRBCR image 
        # (if you just want to focus on luminance values of an image)

        img_rgb = img_rgb.astype(np.float32)
        
        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
        img_ycbcr = img_ycrcb[:,:,(0,2,1)].astype(np.float32)
        img_ycbcr[:,:,0] = (img_ycbcr[:,:,0]*(235-16)+16)/255.0
        img_ycbcr[:,:,1:] = (img_ycbcr[:,:,1:]*(240-16)+16)/255.0

        return img_ycbcr

    def load_image_pair(self, im_id):
        # A method which loads the provided file and returns a numpy array
        # this is used because it will remember in the template dictionary how
        # the images were saved (either as RBG or intensity values or 3D array). 
        # This will help minimize headaches caused by different image types.

        # im_id can either be the index value or the name of the file
        if isinstance(im_id, int):
            HR_file = self.HR_files[im_id]
            LR_file = self.LR_files[im_id]
        elif isinstance(im_id, str):
             _ = self.HR_files.index(im_id)
             HR_file = self.HR_files[_]
             LR_file = self.LR_files[_]
        else:
            TypeError("Invalid image identifier, please input a string to integer")

        im_h = self.load_image(HR_file)
        im_l = self.load_image(LR_file)

        return im_h, im_l


    def save_image(self, fname, im, form=None, verbose = False):
        # Take a given image and save it as the specified format:
        # fname = output name of the saved file
        # im = numpy array of image

        if not form:
            form = self.template['out_type']

        dim = im.shape #Get number of dimensions of image

        if form == 'png':
            # Check that you aren't saving a 3D image
            #TODO: Scale inputs to [0,255] so data isn't lost/image isn't saturated
            cv2.imwrite(f'{fname}',im)
            print(f'Saving: {fname}')
        elif form == 'nii':

            # TODO: Add option to transpose image for some reason because mricron hates the first dim[0] = 1
            # Still gets loaded fine in terms of loading into python, but visualizing it is bad
            # np.transpose(im, (1,2,0))


            # TODO: If image is 2D then append a third  dimension before saving(?)
            nib.save(nib.Nifti1Image(im, np.eye(len(dim)+1)), fname)
            if verbose:
                print(f'Saving: {fname}')
        elif form == 'dcm':
            raise NotImplementedError('DICOM saving currently not supported')
        else:
            raise NotImplementedError('Specified file type is currently not supported for saving')
