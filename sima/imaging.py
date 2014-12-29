"""Base classes for multiframe imaging data."""
import warnings
import collections
import itertools as it
import os
import errno
import csv
from os.path import dirname, join, abspath
import cPickle as pickle
from distutils.version import StrictVersion
from distutils.util import strtobool

import numpy as np
try:
    import h5py
except ImportError:
    h5py_available = False
else:
    h5py_available = StrictVersion(h5py.__version__) >= StrictVersion('2.3.1')

import sima
import sima.misc
from sima.misc import mkdir_p, most_recent_key, affine_transform
from sima.extract import extract_rois, save_extracted_signals
from sima.ROI import ROIList
from sima.neuropil import subtract_neuropil
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sima.misc.tifffile import imsave


class ImagingDataset(object):

    """A multiple sequence imaging dataset.

    Imaging data sets can be iterated over to generate sequences, which
    can in turn be iterated over to generate imaging frames.

    Examples
    --------
    >>> import sima # doctest: +ELLIPSIS
    ...
    >>> from sima.misc import example_data
    >>> dataset = sima.ImagingDataset.load(example_data())

    Datasets can be iterated over as follows:

    >>> for sequence in dataset:
    ...     for frame in sequence:
    ...         for plane in frame:
    ...             for row in plane:
    ...                 for column in row:
    ...                     for channel in column:
    ...                         pass

    Datasets can also be indexed and sliced.

    >>> dataset[0].num_sequences
    1
    >>> dataset[:, 0].num_frames
    1
    >>> dataset[:, 0].frame_shape == dataset.frame_shape
    True

    The resulting sliced datasets are not saved by default.

    Parameters
    ----------
    sequences : list of sima.???.Sequence
        Imaging sequences that can each be iterated over to yield
        the imaging data from each acquistion time.
    savedir : str
        The directory used to store the dataset. If the directory
        name does not end with .sima, then this extension will
        be appended.
    channel_names : list of str, optional
        Names for the channels. Defaults to ['0', '1', '2', ...].
    info : dict
        Data for the order and timing of the data acquisition.
        See Notes for details. *** combine channel names here? ***

    Notes
    -----
    Keys for info:
        'acquisition period' :\n
        'plane order' :\n
        'plane times' :\n
        'plane heights' :\n

    Attributes
    ----------
    num_sequences : int
        The number of sequences in the ImagingDataset.
    frame_shape : tuple of int
        The shape of each frame, in order
        (num_planes, num_rows, num_columns, num_channels).
    num_frames : int
        The total number of image frames in the ImagingDataset.
    ROIs : dict of (str, ROIList)
        The sets of ROIs saved with this ImagingDataset.
    time_averages : list of ndarray
        The time-averaged intensity for each channel.

    """

    def __init__(self, sequences, savedir, channel_names=None, info=None,
                 read_only=False):

        self._read_only = read_only
        self.info = {} if info is None else info
        if sequences is None:
            # Special case used to load an existing ImagingDataset
            if not savedir:
                raise Exception('Cannot initialize dataset without sequences '
                                'or a directory.')

            def unpack(sequence):
                """Parse a saved Sequence dictionary."""
                return sequence.pop('__class__')._from_dict(
                    sequence, savedir)

            with open(join(savedir, 'dataset.pkl'), 'rb') as f:
                data = pickle.load(f)
            self.sequences = [unpack(s) for s in data.pop('sequences')]
            self._channel_names = data.pop('channel_names', None)
            self._savedir = savedir
            self.frame_shape = self.sequences[0].shape[1:]
            try:
                self.num_frames = data.pop('num_frames')
            except KeyError:
                self.num_frames = sum(len(c) for c in self)
        else:
            self.savedir = savedir
            self.sequences = sequences
            self.frame_shape = self.sequences[0].shape[1:]
            if not hasattr(self, 'num_frames'):
                self.num_frames = sum(len(c) for c in self)
            if channel_names is None:
                self.channel_names = [
                    str(x) for x in range(self.frame_shape[-1])]
            else:
                self.channel_names = channel_names
        # initialize sequences
        self.num_sequences = len(self.sequences)
        if not np.all([sequence.shape[1:] == self.sequences[0].shape[1:]
                       for sequence in self.sequences]):
            raise ValueError(
                'All sequences must have images of the same size ' +
                'and the same number of channels.')

    def __getitem__(self, indices):
        if isinstance(indices, int):
            return ImagingDataset([self.sequences[indices]], None,
                                  info=self.info)
        indices = list(indices)
        seq_indices = indices.pop(0)
        if isinstance(seq_indices, int):
            seq_indices = slice(seq_indices, seq_indices + 1)
        sequences = [seq[tuple(indices)] for seq in self.sequences][
            seq_indices]
        return ImagingDataset(sequences, None, info=self.info)

    @property
    def channel_names(self):
        return self._channel_names

    @channel_names.setter
    def channel_names(self, names):
        self._channel_names = [str(n) for n in names]
        if self.savedir is not None and not self._read_only:
            self.save()

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        if savedir is None:
            self._savedir = None
        elif hasattr(self, '_savedir') and savedir == self.savedir:
            return
        else:
            if hasattr(self, '_savedir'):
                orig_dir = self.savedir
            else:
                orig_dir = False
            savedir = abspath(savedir)
            if not savedir.endswith('.sima'):
                savedir += '.sima'
            try:
                os.makedirs(savedir)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(savedir):
                    overwrite = strtobool(
                        raw_input("Overwrite existing directory? "))
                    # Note: This will overwrite dataset.pkl but will leave
                    #       all other files in the directory intact
                    if overwrite:
                        self._savedir = savedir
                    else:
                        return
            else:
                self._savedir = savedir
            if orig_dir:
                from shutil import copy2
                for f in os.listdir(orig_dir):
                    if f.endswith('.pkl'):
                        try:
                            copy2(os.path.join(orig_dir, f), self.savedir)
                        except IOError:
                            pass
            if self._read_only:
                self._read_only = False

    @property
    def time_averages(self):
        if self.savedir is not None:
            try:
                with open(join(self.savedir, 'time_averages.pkl'),
                          'rb') as f:
                    return pickle.load(f)
            except IOError:
                pass
        sums = np.zeros(self.frame_shape)
        counts = np.zeros(self.frame_shape)
        for frame in it.chain.from_iterable(self):
            sums += np.nan_to_num(frame)
            counts[np.isfinite(frame)] += 1
        averages = sums / counts
        if self.savedir is not None:
            with open(join(self.savedir, 'time_averages.pkl'), 'wb') as f:
                pickle.dump(averages, f, pickle.HIGHEST_PROTOCOL)
        return averages

    @property
    def ROIs(self):
        try:
            with open(join(self.savedir, 'rois.pkl'), 'rb') as f:
                return {label: ROIList(**v)
                        for label, v in pickle.load(f).iteritems()}
        except (IOError, pickle.UnpicklingError):
            return {}

    @classmethod
    def load(cls, path):
        """Load a saved ImagingDataset object."""
        try:
            return cls(None, path)
        except (ImportError, KeyError):
            from sima.misc.convert import _load_version0
            # Load a read-only copy of the converted dataset
            ds = _load_version0(path)
            ds._read_only = True
            return ds

    def _todict(self, savedir):
        """Returns the dataset as a dictionary, useful for saving"""
        sequences = [sequence._todict(savedir) for sequence in self]
        return {'sequences': sequences,
                'savedir': abspath(self.savedir),
                'channel_names': self.channel_names,
                'num_frames': self.num_frames,
                '__version__': sima.__version__}

    def add_ROIs(self, ROIs, label=None):
        """Add a set of ROIs to the ImagingDataset.

        Parameters
        ----------
        ROIs : ROIList
            The regions of interest (ROIs) to be added.
        label : str, optional
            The label associated with the ROIList. Defaults to using
            the timestamp as a label.

        Examples
        --------
        Import an ROIList from a zip file containing ROIs created
        with NIH ImageJ.

        >>> from sima.ROI import ROIList
        >>> from sima.misc import example_imagej_rois,example_data
        >>> from sima.imaging import ImagingDataset
        >>> dataset = ImagingDataset.load(example_data())
        >>> rois = ROIList.load(example_imagej_rois(), fmt='ImageJ')
        >>> dataset.add_ROIs(rois, 'from_ImageJ')

        """
        if self.savedir is None:
            raise Exception('Cannot add ROIs unless savedir is set.')
        ROIs.save(join(self.savedir, 'rois.pkl'), label)

    def import_transformed_ROIs(self, source_dataset, source_channel=0,
                                target_channel=0, source_label=None,
                                target_label=None, copy_properties=True):
        """Calculate an affine transformation that maps the source
        ImagingDataset onto this ImagingDataset, tranform the source ROIs
        by this mapping, and then import them into this ImagingDataset.

        Parameters
        ----------
        source_dataset : ImagingDataset
            The ImagingDataset object from which ROIs are to be imported.  This
            dataset must be roughly of the same field-of-view as self in order
            to calculate an affine transformation.

        source_channel : string or int, optional
            The channel of the source image from which to calculate an affine
            transformation, either an integer index or a string in
            source_dataset.channel_names.

        target_channel : string or int, optional
            The channel of the target image from which to calculate an affine
            transformation, either an integer index or a string in
            self.channel_names.

        source_label : string, optional
            The label of the ROIList to transform

        target_label : string, optional
            The label to assign the transformed ROIList

        copy_properties : bool, optional
            Copy the label, id, tags, and im_shape properties from the source
            ROIs to the transformed ROIs
        """

        source_channel = source_dataset._resolve_channel(source_channel)
        target_channel = self._resolve_channel(target_channel)
        source = source_dataset.time_averages[..., source_channel]
        target = self.time_averages[..., target_channel]

        transforms = [affine_transform(s, t) for s, t in
                      it.izip(source, target)]  # zipping over planes

        src_rois = source_dataset.ROIs
        if source_label is None:
            source_label = most_recent_key(src_rois)
        src_rois = src_rois[source_label]
        transformed_ROIs = src_rois.transform(
            transforms, copy_properties=copy_properties)
        self.add_ROIs(transformed_ROIs, label=target_label)

    def delete_ROIs(self, label):
        """Delete an ROI set from the rois.pkl file

        Removes the file if no sets left.

        Parameters
        ----------
        label : string
            The label of the ROI Set to remove from the rois.pkl file

        """

        try:
            with open(join(self.savedir, 'rois.pkl'), 'rb') as f:
                rois = pickle.load(f)
        except IOError:
            return

        try:
            rois.pop(label)
        except KeyError:
            pass
        else:
            if len(rois):
                with open(join(self.savedir, 'rois.pkl'), 'wb') as f:
                    pickle.dump(rois, f, pickle.HIGHEST_PROTOCOL)
            else:
                os.remove(join(self.savedir, 'rois.pkl'))

    def export_averages(self, filenames, fmt='TIFF16', scale_values=True):
        """Save TIFF files with the time average of each channel.

        For datasets with multiple frames, the resulting TIFF files
        have multiple pages.

        Parameters
        ----------
        filenames : str or list of str
            A single (.h5) output filename, or a list of (.tif) output
            filenames with one per channel.
        fmt : {'TIFF8', 'TIFF16'}, optional
            The format of the output files. Defaults to 16-bit TIFF.
        scale_values : bool, optional
            Whether to scale the values to use the full range of the
            output format. Defaults to False.
        """
        if fmt == 'HDF5':
            if not isinstance(filenames, str):
                raise ValueError(
                    'A single filename must be passed for HDF5 format.')
        elif not len(filenames) == self.frame_shape[-1]:
            raise ValueError(
                "The number of filenames must equal the number of channels.")
        for chan, filename in enumerate(filenames):
            im = self.time_averages[:, :, :, chan]
            if dirname(filename):
                mkdir_p(dirname(filename))
            if fmt is 'TIFF8':
                if scale_values:
                    out = sima.misc.to8bit(im)
                else:
                    out = im.astype('uint8')
            elif fmt is 'TIFF16':
                if scale_values:
                    out = sima.misc.to16bit(im)
                else:
                    out = im.astype('uint16')
            else:
                raise ValueError('Unrecognized format.')
            imsave(filename, out)

    def export_frames(self, filenames, fmt='TIFF16', fill_gaps=True,
                      scale_values=False):
        """Save a multi-page TIFF files of the motion-corrected time series.

        # TODO: HDF5, multiple Z planes
        One TIFF file is created for each sequence and channel.
        The TIFF files have the same name as the uncorrected files, but should
        be saved in a different directory.

        Parameters
        ----------
        filenames : list of list of list of string or list of string
            Path to the locations where the output files will be saved.
            If fmt is TIFF, filenames[i][j][k] is the path to the file
            for sequence i, plane j, chanel k.  If fmt is 'HDF5', filenames[i]
            is the path to the file for the ith sequence.
        fmt : {'TIFF8', 'TIFF16', 'HDF5'}, optional
            The format of the output files. Defaults to 16-bit TIFF.
        fill_gaps : bool, optional
            Whether to fill in unobserved rows with data from adjacent frames.
            Defaults to True.
        scale_values : bool, optional
            Whether to scale the values to use the full range of the
            output format. Defaults to False.
        """
        depth = lambda L: \
            isinstance(L, collections.Sequence) and \
            (not isinstance(L, str)) and max(map(depth, L)) + 1
        if (fmt in ['TIFF16', 'TIFF8']) and not depth(filenames) == 3:
            raise ValueError
        if fmt == 'HDF5' and not depth(filenames) == 1:
            raise ValueError
        for sequence, fns in it.izip(self, filenames):
            sequence.export(fns, fmt, fill_gaps, self.channel_names)

    def export_signals(self, path, fmt='csv', channel=0, signals_label=None):
        """Export extracted signals to a file.

        Parameters
        ----------
        path : str
            The name of the file that will store the exported data.
        fmt : {'csv'}, optional
            The export format. Currently, only 'csv' export is available.
        channel : string or int
            The channel from which to export signals, either an integer
            index or a string in self.channel_names.
        signals_label : str, optional
            The label of the extracted signal set to use. By default,
            the most recently extracted signals are used.
        """
        with open(path, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            signals = self.signals(channel)
            if signals_label is None:
                signals_label = most_recent_key(signals)
            rois = signals[signals_label]['rois']

            writer.writerow(['sequence', 'frame'] + [r['id'] for r in rois])
            writer.writerow(['', 'label'] + [r['label'] for r in rois])
            writer.writerow(
                ['', 'tags'] +
                [''.join(t + ',' for t in sorted(r['tags']))[:-1]
                 for r in rois]
            )
            for sequence_idx, sequence in enumerate(
                    signals[signals_label]['raw']):
                for frame_idx, frame in enumerate(sequence.T):
                    writer.writerow([sequence_idx, frame_idx] + frame.tolist())

    def extract(self, rois=None, signal_channel=0, label=None,
                remove_overlap=True, n_processes=None, demix_channel=None,
                save_summary=True):
        """Extracts imaging data from the current dataset using the
        supplied ROIs file.

        Parameters
        ----------
        rois : sima.ROI.ROIList, optional
            ROIList of rois to extract
        signal_channel : string or int, optional
            Channel containing the signal to be extracted, either an integer
            index or a name in self.channel_names
        label : string or None, optional
            Text label to describe this extraction, if None defaults to a
            timestamp.
        remove_overlap : bool, optional
            If True, remove any pixels that overlap between masks.
        n_processes : int, optional
            Number of processes to farm out the extraction across. Should be
            at least 1 and at most one less then the number of CPUs in the
            computer. If None, uses half the CPUs.
        demix_channel : string or int, optional
            Channel to demix from the signal channel, either an integer index
            or a name in self.channel_names If None, do not demix signals.
        save_summary : bool, optional
            If True, additionally save a summary of the extracted ROIs.

        Return
        ------
        dict of arrays
            Keys: raw, demixed_raw, mean_frame, overlap, signal_channel, rois,
            timestamp

        See also
        --------
        sima.ROI.ROIList

        """

        signal_channel = self._resolve_channel(signal_channel)
        demix_channel = self._resolve_channel(demix_channel)

        if rois is None:
            rois = self.ROIs[most_recent_key(self.ROIs)]
        if rois is None or not len(rois):
            raise Exception('Cannot extract dataset with no ROIs.')
        if self.savedir:
            return save_extracted_signals(
                self, rois, self.savedir, label, signal_channel=signal_channel,
                remove_overlap=remove_overlap, n_processes=n_processes,
                demix_channel=demix_channel, save_summary=save_summary
            )
        else:
            return extract_rois(self, rois, signal_channel, remove_overlap,
                                n_processes, demix_channel)

    def subtract_neuropil(self, channel=0, label=None, subtraction_kwargs={}):
        """Apply a neuropil subtraction algorithm to a signals set

        Parameters
        ----------
        channel : string or int
            The channel from which to subtract the neuropil signal.  The
            neuropil signal is taken from this channel as well.
        label : string
            The label of the signals to subtract
        subtraction_kwargs : dict
            Keyword arguments accepted by the subtraction method

        """
        channel = self._resolve_channel(channel)
        signals = self.signals(channel=channel)
        if label is None:
            label = most_recent_key(signals)

        signals[label]['subtracted'] = subtract_neuropil(
            self, channel, label, subtraction_kwargs)

        with open(join(self.savedir, 'signals_{}.pkl'.format(channel)), 'wb') as f:
            pickle.dump(signals, f, pickle.HIGHEST_PROTOCOL)

    def save(self, savedir=None):
        """Save the ImagingDataset to a file."""

        if savedir is None:
            savedir = self.savedir
        self.savedir = savedir

        if self._read_only:
            raise Exception('Cannot save read-only dataset.  Change savedir ' +
                            'to a new directory')
        with open(join(savedir, 'dataset.pkl'), 'wb') as f:
            pickle.dump(self._todict(savedir), f, pickle.HIGHEST_PROTOCOL)

    def segment(self, method, label=None, planes=None):
        """Segment an ImagingDataset to generate ROIs.

        Parameters
        ----------
        method : sima.segment.SegmentationStrategy
            The method for segmentation. Defaults to normcut.
        label : str, optional
            Label to be associated with the segmented set of ROIs.
        planes : list of int
            List of the planes that are to be segmented.

        Returns
        -------
        ROIs : sima.ROI.ROIList
            The segmented regions of interest.
        """
        rois = method.segment(self)
        if self.savedir is not None:
            rois.save(join(self.savedir, 'rois.pkl'), label)
        return rois

    def signals(self, channel=0):
        """Return a dictionary of extracted signals

        Parameters
        ----------
        channel : string or int
            The channel to load signals for, either an integer index or a
            string in self.channel_names

        """
        channel = self._resolve_channel(channel)
        try:
            with open(join(self.savedir, 'signals_{}.pkl'.format(channel)),
                      'rb') as f:
                return pickle.load(f)
        except (IOError, pickle.UnpicklingError):
            return {}

    def __str__(self):
        return '<ImagingDataset>'

    def __repr__(self):
        return ('<ImagingDataset: ' + 'num_sequences={n_sequences}, ' +
                'frame_shape={fsize}, num_frames={frames}>').format(
            n_sequences=self.num_sequences,
            fsize=self.frame_shape,
            frames=self.num_frames)

    def __iter__(self):
        return self.sequences.__iter__()

    def _resolve_channel(self, chan):
        """Return the index corresponding to the channel."""
        return sima.misc.resolve_channels(chan, self.channel_names)
