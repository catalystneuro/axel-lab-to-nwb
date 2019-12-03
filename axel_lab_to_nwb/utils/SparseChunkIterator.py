# Function to select chunks to write to disk

from hdmf.data_utils import AbstractDataChunkIterator, DataChunk
import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.measure import block_reduce


class SparseIterator(AbstractDataChunkIterator):

    def __init__(self, data, chunk_shape):
        """ Select chunks to write to disk
        Params:
          data:              The data to write to disk
          shape:             shape of data
          chunk_shape:       The shape of each chunk to be created
          chunks_write:      List of selected chunks to write
          chunk_count:       number of selected chunks to write
          chunk_ratio:       ratio of selected chunks to total number of chunks in the data
        Returns:
          Sparse data chunk iterator
        """

        self.shape, self.chunk_shape = data.shape, chunk_shape
        self.data = data

        self.__chunks_created = 0

        self.chunks_write, self.chunk_maxvalues = self.blob_detection()
        self.chunk_count = len(self.chunks_write)
        self.chunk_ratio = self.chunk_count/np.prod( np.ceil( np.divide( self.shape, self.chunk_shape ) ).astype(int) )

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return in each iteration a fully occupied data chunk of self.chunk_shape values at selected
        location within the data
        """

        if self.__chunks_created < self.chunk_count:

            chunk_i = self.__chunks_created
            chunk_index = self.chunks_write[ chunk_i ]
            tmin, xmin, ymin, zmin = chunk_index*self.chunk_shape
            tmax, xmax, ymax, zmax = chunk_index*self.chunk_shape + self.chunk_shape
            tmax, xmax, ymax, zmax = np.clip( [tmax, xmax, ymax, zmax], np.zeros( len(self.shape), dtype=int ), self.shape )

            selection = np.s_[tmin:tmax, xmin:xmax, ymin:ymax, zmin:zmax]
            data = self.data[selection]
            self.__chunks_created += 1

            return DataChunk(data=data, selection=selection)
        else:
            raise StopIteration

    next = __next__

    def recommended_chunk_shape(self):
        # Here we can optionally recommend what a good chunking should be.
        return self.chunk_shape

    def recommended_data_shape(self):
        # In cases where we don't know the full size this should be the minimum size.
        return (1,*self.shape[1:])

    @property
    def dtype(self):
        # The data type of our array
        return np.dtype(np.float32)

    @property
    def maxshape(self):
        # If we don't know the size of a dimension beforehand we can set the dimension to None instead
        return (None,*self.shape[1:])

    # Defined only to allow for passing data through if chunk_shape is None
    def __new__(cls, data, chunk_shape):
        if chunk_shape != None:
            #Create instance and call SparseIterator init 
            return super().__new__(cls)
        else:
            #pass data through
            return data

    def blob_detection(self, scale = 1.5, threshold = 1):
        """ Detect chunks of interest
        Params:
          scale:     scale of the gaussian filter
          threshold: threshold for detecting data
        Returns:
            List of selected chunks to write
        """

        # find maximum across time dimension
        data_max = np.max(self.data, axis = 0)
        data_gaussian_laplace = scale*gaussian_laplace(data_max, sigma=scale, mode='nearest')
        # Finding laplacian's max values in each chunk
        chunk_maxvalues = block_reduce(data_gaussian_laplace, block_size=self.chunk_shape[1:], func=np.max)
        # repeating chunk_maxvalues across time dimension
        chunk_repeat = np.ones( len(self.shape) )
        chunk_repeat[0] = np.ceil( self.shape[0]/self.chunk_shape[0] )
        chunk_maxvalues_tiled = np.tile( chunk_maxvalues, chunk_repeat.astype(int) )
        chunks_write = np.argwhere( chunk_maxvalues_tiled > threshold )

        return chunks_write, chunk_maxvalues

